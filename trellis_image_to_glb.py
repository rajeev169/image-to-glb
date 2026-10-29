#!/usr/bin/env python3
"""
Image -> GLB using a free TRELLIS Space on Hugging Face (no local GPU needed).

Usage:
    python trellis_image_to_glb.py rider_white_bg.png -o output/model.glb

Optional: set HF_TOKEN (free Hugging Face account) for a bigger GPU quota.
The script reads each Space's API at run time, so small API changes don't break it.
"""
import argparse
import datetime as dt
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from gradio_client import Client, handle_file

from glb_utils import check_glb, fix_glb_material

# Tried in order. Spaces can be busy, broken or renamed, so we fall back.
SPACES = ["microsoft/TRELLIS.2", "JeffreyXiang/TRELLIS", "microsoft/TRELLIS"]

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


class QuotaError(RuntimeError):
    """Free GPU time on Hugging Face is used up. Retrying now only wastes time."""


class SpaceBroken(RuntimeError):
    """The Space itself is down (e.g. CONFIG_ERROR). Only its owner can fix it."""


def classify(err: Exception) -> Exception:
    text = str(err)
    low = text.lower()
    if "gpu quota" in low or "zerogpu" in low:
        return QuotaError(text)
    if "invalid state" in low or "config_error" in low or "runtime_error" in low:
        return SpaceBroken(text)
    return err


def quota_message(text: str) -> str:
    """Turn 'Try again in 23:48:47' into a clear wait time in India time."""
    m = re.search(r"try again in (\d+):(\d{2}):(\d{2})", text, re.I)
    if not m:
        return "Free Hugging Face GPU quota is used up. Try again later."
    h, mi, s = (int(x) for x in m.groups())
    wait = dt.timedelta(hours=h, minutes=mi, seconds=s)
    at = (dt.datetime.now(IST) + wait).strftime("%d %b, %I:%M %p IST")
    return f"Free Hugging Face GPU quota is used up. Wait about {h} h {mi} min (until {at}), or use the Kaggle option."


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


def call(client: Client, api_name: str, provided: dict, image=None):
    """Call an endpoint with only the arguments it really has.
    `image` goes to the endpoint's first argument, whatever that Space calls it
    (e.g. 'image' in one Space, 'input' in another).
    Arguments we don't know about use the Space's own defaults."""
    params = params_of(client, api_name)
    provided = dict(provided)
    if image is not None and params:
        provided.setdefault(params[0].get("parameter_name"), image)
    kwargs = {}
    for p in params:
        name = p.get("parameter_name")
        if name in provided:
            kwargs[name] = provided[name]
        elif not p.get("parameter_has_default", False):
            raise RuntimeError(f"{api_name} needs an argument '{name}' this script doesn't know about")
    return client.predict(api_name=api_name, **kwargs)


def has_transparency(image: str) -> bool:
    from PIL import Image
    with Image.open(image) as im:
        if im.mode not in ("RGBA", "LA") and "transparency" not in im.info:
            return False
        alpha = im.convert("RGBA").getchannel("A")
        return alpha.getextrema()[0] < 250


def remove_bg_local(image: str) -> str:
    """Remove the background on the GitHub machine (CPU) with rembg. Returns a transparent PNG."""
    from PIL import Image
    from rembg import remove
    out = Path(tempfile.gettempdir()) / "input_no_bg.png"
    with Image.open(image) as im:
        remove(im.convert("RGB")).save(out)
    return str(out)


def generate(client: Client, image: str, args) -> str:
    try:
        client.predict(api_name="/start_session")
    except Exception:
        pass  # not all versions have it

    img = image
    if not args.no_preprocess:
        try:
            img = as_path(call(client, "/preprocess_image", {}, image=handle_file(image)))
            print("  background removed by the Space")
        except Exception as e:
            err = classify(e)
            if isinstance(err, (QuotaError, SpaceBroken)):
                raise err
            print(f"  (Space background removal failed: {e})")
            if has_transparency(image):
                img = image
            else:
                print("  removing background locally with rembg...")
                img = remove_bg_local(image)

    print("  generating 3D (this takes 1-3 minutes)...")
    call(client, "/image_to_3d", {
        "image": handle_file(img),
        "seed": args.seed,
        "randomize_seed": False,
        "multiimages": [],
        "multiimage_algo": "stochastic",
    }, image=handle_file(img))

    print("  extracting GLB...")
    provided = {"texture_size": args.texture}
    if args.triangles:
        provided["decimation_target"] = args.triangles
    provided["mesh_simplify"] = args.simplify  # older TRELLIS Spaces use this name
    return as_path(call(client, "/extract_glb", provided))


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

    spaces = [args.space] if args.space else list(SPACES)

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
        print("Give the image path as it is in the repo, e.g. rider_white_bg.png")
        return 1
    if not os.environ.get("HF_TOKEN"):
        print("Note: HF_TOKEN is not set. GitHub machines share one tiny anonymous GPU quota, so add a free token.")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    broken = set()
    last_err = None
    for attempt in range(1, args.retries + 1):
        for space in spaces:
            if space in broken:
                continue
            try:
                print(f"[{attempt}/{args.retries}] using {space}")
                client = connect(space)
                glb = generate(client, args.image, args)
                shutil.copy(glb, out)
                check_glb(out)
                if not args.no_fix_material:
                    fix_glb_material(out)
                print(f"Done: {out} ({out.stat().st_size / 1e6:.1f} MB) from {space}")
                return 0
            except Exception as e:
                err = classify(e)
                last_err = err
                if isinstance(err, QuotaError):
                    print(f"  failed: {err}")
                    print(quota_message(str(err)))
                    return 3
                if isinstance(err, SpaceBroken):
                    print(f"  {space} is down (only its owner can fix it), skipping it from now on")
                    broken.add(space)
                    continue
                print(f"  failed: {err}")
        if len(broken) == len(spaces):
            print("Every Space is down right now. Use the Kaggle option instead.")
            return 2
        if attempt < args.retries:
            wait = 30 * attempt
            print(f"Waiting {wait}s before retrying (queue)...")
            time.sleep(wait)

    print(f"All attempts failed. Last error: {last_err}")
    print("Tip: run with --list-api to see each Space's API, and set HF_TOKEN for more GPU quota.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
