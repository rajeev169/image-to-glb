#!/usr/bin/env python3
"""
Run TRELLIS on a free NVIDIA GPU (Kaggle or Google Colab) - no Hugging Face quota.

    python gpu/trellis_gpu.py install                   # 15-25 min, once per session
    python gpu/trellis_gpu.py generate image.png -o model.glb

Needs: NVIDIA GPU with 16 GB (T4 is fine), internet on.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # for glb_utils.py

TRELLIS_DIR = Path(os.environ.get("TRELLIS_DIR", "/tmp/TRELLIS"))
EXT_DIR = Path("/tmp/extensions")
DONE_FLAG = TRELLIS_DIR / ".install_done"

# Versions TRELLIS was built and tested with. Fixed so a new Kaggle/Colab image can't break it.
TORCH = ["torch==2.4.0", "torchvision==0.19.0", "--index-url", "https://download.pytorch.org/whl/cu121"]
XFORMERS = ["xformers==0.0.27.post2", "--index-url", "https://download.pytorch.org/whl/cu121"]
BASIC = [
    "numpy<2", "pillow", "imageio", "imageio-ffmpeg", "tqdm", "easydict", "opencv-python-headless<4.11",
    "scipy", "ninja", "rembg", "onnxruntime", "trimesh", "xatlas", "pyvista", "pymeshfix", "igraph",
    "transformers<4.50", "huggingface_hub", "safetensors", "spconv-cu120",
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
]


def run(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def pip(*args):
    run([sys.executable, "-m", "pip", "install", "-q", *args])


def cuda_env() -> dict:
    env = dict(os.environ)
    for cuda in ("/usr/local/cuda", "/usr/local/cuda-12"):
        if Path(cuda, "bin", "nvcc").exists():
            env["CUDA_HOME"] = cuda
            env["PATH"] = f"{cuda}/bin:" + env.get("PATH", "")
            break
    try:
        cap = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True
        ).split()[0]
        env["TORCH_CUDA_ARCH_LIST"] = cap  # build only for this GPU = much faster
    except Exception:
        env["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;8.9"
    env["FORCE_CUDA"] = "1"
    env["MAX_JOBS"] = env.get("MAX_JOBS", "4")
    return env


def check_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
        ).strip()
    except Exception:
        sys.exit("No NVIDIA GPU found. Kaggle: Settings > Accelerator > GPU T4. Colab: Runtime > Change runtime type > T4 GPU.")
    print("GPU:", out.splitlines()[0])


def install():
    check_gpu()
    if DONE_FLAG.exists():
        print("TRELLIS already installed.")
        return
    env = cuda_env()
    print("Installing PyTorch 2.4 + TRELLIS dependencies (15-25 minutes)...")
    pip(*TORCH)
    pip(*XFORMERS)
    pip(*BASIC)

    if not (TRELLIS_DIR / "trellis").exists():
        shutil.rmtree(TRELLIS_DIR, ignore_errors=True)
        run(["git", "clone", "--depth", "1", "https://github.com/microsoft/TRELLIS.git", str(TRELLIS_DIR)])

    EXT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building GPU extensions (nvdiffrast, gaussian rasterizer)...")
    nvd = EXT_DIR / "nvdiffrast"
    if not nvd.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/NVlabs/nvdiffrast.git", str(nvd)])
    run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", str(nvd)], env=env)

    mip = EXT_DIR / "mip-splatting"
    if not mip.exists():
        run(["git", "clone", "--depth", "1", "--recurse-submodules",
             "https://github.com/autonomousvision/mip-splatting.git", str(mip)])
    run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation",
         str(mip / "submodules" / "diff-gaussian-rasterization")], env=env)

    # Quick import test so a broken install fails here, not after the model download.
    run([sys.executable, "-c",
         "import torch, xformers, spconv, nvdiffrast.torch, diff_gaussian_rasterization, utils3d;"
         "assert torch.cuda.is_available(), 'CUDA not available';"
         "print('torch', torch.__version__, 'cuda ok')"])
    DONE_FLAG.write_text("ok")
    print("Install finished.")


def generate(image: str, output: str, seed: int, texture: int, simplify: float):
    os.environ.setdefault("ATTN_BACKEND", "xformers")  # flash-attn doesn't run on T4
    os.environ.setdefault("SPCONV_ALGO", "native")
    sys.path.insert(0, str(TRELLIS_DIR))

    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils
    from glb_utils import check_glb, fix_glb_material

    print("Loading TRELLIS-image-large (first time downloads ~3 GB)...", flush=True)
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    print("Generating 3D...", flush=True)
    img = Image.open(image)
    outputs = pipeline.run(img, seed=seed, formats=["gaussian", "mesh"], preprocess_image=True)

    # Save the background-removed image too, so you can see what TRELLIS actually used.
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        pipeline.preprocess_image(Image.open(image)).save(out.with_name("input_used.png"))
    except Exception:
        pass

    print("Building GLB (texture baking takes 1-2 minutes)...", flush=True)
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0], outputs["mesh"][0],
        simplify=simplify, texture_size=texture, verbose=False,
    )
    glb.export(str(out))
    check_glb(out)
    fix_glb_material(out)
    print(f"Done: {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("install")
    g = sub.add_parser("generate")
    g.add_argument("image")
    g.add_argument("-o", "--output", default="output/model.glb")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--texture", type=int, default=1024)
    g.add_argument("--simplify", type=float, default=0.95)
    a = ap.parse_args()
    if a.cmd == "install":
        install()
    else:
        if not DONE_FLAG.exists():
            sys.exit("Run 'python gpu/trellis_gpu.py install' first.")
        generate(a.image, a.output, a.seed, a.texture, a.simplify)


if __name__ == "__main__":
    main()
