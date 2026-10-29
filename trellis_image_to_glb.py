#!/usr/bin/env python3
"""
Image -> GLB using the free TRELLIS Space on Hugging Face (no local GPU needed).

Usage:
    python trellis_image_to_glb.py input/rider.png -o output/rider.glb

Optional: set HF_TOKEN (free Hugging Face account) for a bigger GPU quota.
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

# Tried in order. Spaces can be busy, asleep or renamed, so we fall back.
SPACES = ["JeffreyXiang/TRELLIS", "microsoft/TRELLIS"]


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


def generate(client: Client, image: str, seed: int, simplify: float, texture_size: int,
             preprocess: bool, ss_steps: int, slat_steps: int) -> str:
    try:
        client.predict(api_name="/start_session")
    except Exception:
        pass  # not all versions have it

    img = image
    if preprocess:
        try:
            img = as_path(client.predict(image=handle_file(image), api_name="/preprocess_image"))
        except Exception as e:
            print(f"  (preprocess skipped: {e})")
            img = image

    print("  generating 3D (this takes 1-3 minutes)...")
    client.predict(
        image=handle_file(img),
        multiimages=[],
        seed=seed,
        ss_guidance_strength=7.5,
        ss_sampling_steps=ss_steps,
        slat_guidance_strength=3,
        slat_sampling_steps=slat_steps,
        multiimage_algo="stochastic",
        api_name="/image_to_3d",
    )

    print("  extracting GLB...")
    result = client.predict(mesh_simplify=simplify, texture_size=texture_size, api_name="/extract_glb")
    return as_path(result)


def fix_glb_material(path: Path, metallic: float = 0.0, roughness: float = 0.85) -> None:
    """TRELLIS GLBs leave 'metallic' unset, which viewers treat as fully metal (looks black).
    Set it to non-metal so the model shows its real colours."""
    data = path.read_bytes()
    if data[:4] != b"glTF":
        return
    total = struct.unpack("<I", data[8:12])[0]
    jlen, jtype = struct.unpack("<I4s", data[12:20])
    if jtype != b"JSON":
        return
    gltf = json.loads(data[20:20 + jlen])
    rest = data[20 + jlen:total]
    for mat in gltf.get("materials", []):
        pbr = mat.setdefault("pbrMetallicRoughness", {})
        pbr["metallicFactor"] = metallic
        pbr["roughnessFactor"] = roughness
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    out = b"glTF" + struct.pack("<II", 2, 20 + len(js) + len(rest))
    out += struct.pack("<I4s", len(js), b"JSON") + js + rest
    path.write_bytes(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert an image to a GLB with TRELLIS (Hugging Face Space).")
    ap.add_argument("image", help="input image (png/jpg). Plain background works best.")
    ap.add_argument("-o", "--output", default="output/model.glb")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--simplify", type=float, default=0.95, help="mesh simplify 0.90-0.98 (higher = fewer triangles)")
    ap.add_argument("--texture", type=int, default=1024, choices=[512, 1024, 1536, 2048])
    ap.add_argument("--steps", type=int, default=12, help="sampling steps (12 is the default)")
    ap.add_argument("--no-preprocess", action="store_true", help="skip background removal")
    ap.add_argument("--no-fix-material", action="store_true", help="keep the original metallic setting")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    if not Path(args.image).exists():
        print(f"Input not found: {args.image}")
        return 1
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    last_err = None
    for attempt in range(1, args.retries + 1):
        for space in SPACES:
            try:
                print(f"[{attempt}/{args.retries}] using {space}")
                client = connect(space)
                glb = generate(client, args.image, args.seed, args.simplify, args.texture,
                               not args.no_preprocess, args.steps, args.steps)
                shutil.copy(glb, out)
                if not args.no_fix_material:
                    fix_glb_material(out)
                print(f"Done: {out} ({out.stat().st_size / 1e6:.1f} MB)")
                return 0
            except Exception as e:
                last_err = e
                print(f"  failed: {e}")
        if attempt < args.retries:
            wait = 30 * attempt
            print(f"Waiting {wait}s before retrying (queue or GPU quota)...")
            time.sleep(wait)

    print(f"All attempts failed. Last error: {last_err}")
    print("Tip: run client.view_api() on the Space to see if its API changed, or set HF_TOKEN.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
