#!/usr/bin/env python3
"""
Image -> GLB using a free TRELLIS Space on Hugging Face (no local GPU needed).

Usage:
    python trellis_image_to_glb.py rider_white_bg.png -o output/model.glb

Optional: set HF_TOKEN (free Hugging Face account) for a bigger GPU quota.
The script reads each Space's API at run time, so small API changes don't break it.
"""
import argparse
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

from gradio_client import Client, handle_file

# Tried in order. Spaces can be busy, broken or renamed, so we fall back.
SPACES = ["microsoft/TRELLIS.2", "JeffreyXiang/TRELLIS", "microsoft/TRELLIS"]


def connect(space: str) -> Client:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return Client(space)
    try:
        return Client(space, token=token)
    except TypeError:  # older gradio_client versions
        return Client(space, hf_token=token)


def as_path(value) -> str:
    """Gradio returns files as str, dict or tuple depending on version. Pull out the file path."""
    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            try:
                return as_path(item)
            except ValueError:
                continue
        raise ValueError("no file path in result")
    if isinstance(value, dict):
        for key in ("path", "value", "video", "name"):
            if key in value and value[key]:
                return as_path(value[key])
        raise ValueError("no file path in result")
    if isinstance(value, str) and os.path.exists(value):
        return value
    raise ValueError(f"not a file path: {value!r}")


def params_of(client: Client, api_name: str):
    info = client.view_api(return_format="dict", print_info=False)
    endpoints = info.get("named_endpoints", {})
    if api_name not in endpoints:
        raise RuntimeError(f"{api_name} not found. Available: {list(endpoints)}")
    return endpoints[api_name].get("parameters", [])


def call(client: Client, api_name: str, provided: dict):
    """Call an endpoint with only the arguments it really has.
    Arguments we don't know about use the Space's own defaults."""
    kwargs = {}
    for p in params_of(client, api_name):
        name = p.get("parameter_name")
        if name in provided:
            kwargs[name] = provided[name]
        elif not p.get("parameter_has_default", False):
            raise RuntimeError(f"{api_name} needs an argument '{name}' this script doesn't know about")
    return client.predict(api_name=api_name, **kwargs)


def generate(client: Client, image: str, args) -> str:
    try:
        client.predict(api_name="/start_session")
    except Exception:
        pass  # not all versions have it

    img = image
    if not args.no_preprocess:
        try:
            img = as_path(call(client, "/preprocess_image", {"image": handle_file(image)}))
        except Exception as e:
            print(f"  (preprocess skipped: {e})")
            img = image

    print("  generating 3D (this takes 1-3 minutes)...")
    call(client, "/image_to_3d", {
        "image": handle_file(img),
        "seed": args.seed,
        "randomize_seed": False,
        "multiimages": [],
        "multiimage_algo": "stochastic",
    })

    print("  extracting GLB...")
    provided = {"texture_size": args.texture}
    if args.triangles:
        provided["decimation_target"] = args.triangles
    provided["mesh_simplify"] = args.simplify  # older TRELLIS Spaces use this name
    return as_path(call(client, "/extract_glb", provided))


def fix_glb_material(path: Path, metallic: float = 0.0, roughness: float = 0.85) -> None:
    """Old TRELLIS GLBs leave 'metallic' unset, which viewers treat as full metal (looks black).
    Set it to non-metal. Models that already have a metallic/roughness texture are left alone."""
    data = path.read_bytes()
    if data[:4] != b"glTF":
        return
    total = struct.unpack("<I", data[8:12])[0]
    jlen, jtype = struct.unpack("<I4s", data[12:20])
    if jtype != b"JSON":
        return
    gltf = json.loads(data[20:20 + jlen])
    rest = data[20 + jlen:total]
    changed = False
    for mat in gltf.get("materials", []):
        pbr = mat.setdefault("pbrMetallicRoughness", {})
        if "metallicRoughnessTexture" in pbr or "metallicFactor" in pbr:
            continue
        pbr["metallicFactor"] = metallic
        pbr["roughnessFactor"] = roughness
        changed = True
    if not changed:
        return
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    out = b"glTF" + struct.pack("<II", 2, 20 + len(js) + len(rest))
    out += struct.pack("<I4s", len(js), b"JSON") + js + rest
    path.write_bytes(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert an image to a GLB with TRELLIS (Hugging Face Space).")
    ap.add_argument("image", help="input image (png/jpg). Plain background works best.")
    ap.add_argument("-o", "--output", default="output/model.glb")
    ap.add_argument("--space", help="use only this Space, e.g. microsoft/TRELLIS.2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--texture", type=int, default=1024, help="texture size: 512, 1024 or 2048")
    ap.add_argument("--triangles", type=int, default=0, help="target triangle count (TRELLIS.2 Space only)")
    ap.add_argument("--simplify", type=float, default=0.95, help="mesh simplify 0.90-0.98 (old TRELLIS Space only)")
    ap.add_argument("--no-preprocess", action="store_true", help="skip background removal")
    ap.add_argument("--no-fix-material", action="store_true", help="keep the original metallic setting")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--list-api", action="store_true", help="print each Space's API and exit")
    args = ap.parse_args()

    spaces = [args.space] if args.space else SPACES

    if args.list_api:
        for space in spaces:
            print(f"==== {space}")
            try:
                connect(space).view_api()
            except Exception as e:
                print(f"  failed: {e}")
        return 0

    if not Path(args.image).exists():
        print(f"Input not found: {args.image}")
        return 1
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    last_err = None
    for attempt in range(1, args.retries + 1):
        for space in spaces:
            try:
                print(f"[{attempt}/{args.retries}] using {space}")
                client = connect(space)
                glb = generate(client, args.image, args)
                shutil.copy(glb, out)
                if not args.no_fix_material:
                    fix_glb_material(out)
                print(f"Done: {out} ({out.stat().st_size / 1e6:.1f} MB) from {space}")
                return 0
            except Exception as e:
                last_err = e
                print(f"  failed: {e}")
        if attempt < args.retries:
            wait = 30 * attempt
            print(f"Waiting {wait}s before retrying (queue or GPU quota)...")
            time.sleep(wait)

    print(f"All attempts failed. Last error: {last_err}")
    print("Tip: run with --list-api to see each Space's API, and set HF_TOKEN for more GPU quota.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
