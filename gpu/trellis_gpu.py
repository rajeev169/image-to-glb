#!/usr/bin/env python3
"""
Run TRELLIS on a free NVIDIA GPU (Kaggle or Google Colab) - no Hugging Face quota.

    python gpu/trellis_gpu.py install                   # 15-25 min, once per session
    python gpu/trellis_gpu.py generate image.png -o model.glb

Needs: NVIDIA GPU with 16 GB (T4 is fine), internet on.

Kaggle/Colab change their Python version often (Kaggle is on 3.13 now), but TRELLIS
needs PyTorch 2.4, which only runs on Python <= 3.12. So everything is installed into
its own Python 3.10 environment (made with `uv`) and both steps run inside it.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent.parent))  # for glb_utils.py

TRELLIS_DIR = Path(os.environ.get("TRELLIS_DIR", "/tmp/TRELLIS"))
EXT_DIR = Path("/tmp/extensions")
VENV = Path(os.environ.get("TRELLIS_VENV", "/tmp/trellis-venv"))
VENV_PY = VENV / "bin" / "python"
DONE_FLAG = VENV / ".install_done"
PY_VERSION = "3.10"

# Versions TRELLIS was built and tested with. Fixed so a new Kaggle/Colab image can't break it.
TORCH = ["torch==2.4.0", "torchvision==0.19.0", "--index-url", "https://download.pytorch.org/whl/cu121"]
XFORMERS = ["xformers==0.0.27.post2", "--index-url", "https://download.pytorch.org/whl/cu121"]
BASIC = [
    "numpy<2", "pillow", "imageio", "imageio-ffmpeg", "tqdm", "easydict", "opencv-python-headless<4.11",
    "scipy", "ninja", "setuptools<70", "wheel", "rembg", "onnxruntime", "trimesh", "xatlas", "pyvista",
    "pymeshfix", "igraph", "plyfile", "open3d==0.18.0", "transformers<4.50", "huggingface_hub", "safetensors", "spconv-cu120",
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
]

NO_GPU_HELP = (
    "No NVIDIA GPU in this session.\n"
    "  Kaggle: the account must be phone-verified (kaggle.com > Settings > Phone verification),\n"
    "          and weekly GPU hours must be left. Notebook: Settings > Accelerator > GPU T4 x2.\n"
    "  Colab:  Runtime > Change runtime type > T4 GPU."
)


def run(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def in_venv() -> bool:
    return Path(sys.executable).resolve() == VENV_PY.resolve() if VENV_PY.exists() else False


def uv_bin() -> str:
    found = os.environ.get("TRELLIS_UV") or shutil.which("uv")
    if found:
        return found
    for cand in (Path(sys.executable).parent / "uv", Path.home() / ".local" / "bin" / "uv"):
        if cand.exists():
            return str(cand)
    raise RuntimeError("uv not found")


def pip(*args, env=None):
    # Inside the venv we use uv for speed; it installs into this interpreter.
    run([uv_bin(), "pip", "install", "--python", sys.executable, *args], env=env)


def gpu_info():
    for exe in ("nvidia-smi", "/usr/bin/nvidia-smi", "/usr/local/nvidia/bin/nvidia-smi"):
        try:
            out = subprocess.check_output(
                [exe, "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"], text=True
            ).strip()
            if out:
                return [x.strip() for x in out.splitlines()[0].split(",")]
        except Exception:
            continue
    return None


def cuda_env(cap: str) -> dict:
    env = dict(os.environ)
    for cuda in ("/usr/local/cuda", "/usr/local/cuda-12"):
        if Path(cuda, "bin", "nvcc").exists():
            env["CUDA_HOME"] = cuda
            env["PATH"] = f"{cuda}/bin:" + env.get("PATH", "")
            break
    env["TORCH_CUDA_ARCH_LIST"] = cap or "7.5;8.0;8.6;8.9"  # build only for this GPU = much faster
    env["FORCE_CUDA"] = "1"
    env.setdefault("MAX_JOBS", "4")
    return env


def make_venv():
    """Create a Python 3.10 environment with uv (downloads its own Python if needed)."""
    if VENV_PY.exists():
        return
    try:
        uv_bin()
    except RuntimeError:
        run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    run([uv_bin(), "venv", "--seed", "-p", PY_VERSION, str(VENV)])


def install():
    info = gpu_info()
    if not info:
        sys.exit(NO_GPU_HELP)
    print("GPU:", ", ".join(info), flush=True)

    if not in_venv():
        if DONE_FLAG.exists():
            print("TRELLIS already installed.")
            return
        make_venv()
        env = dict(os.environ, TRELLIS_UV=uv_bin())  # same uv inside the venv step
        run([VENV_PY, "-u", HERE, "install"], env=env)
        return

    # ---------------- inside the Python 3.10 venv ----------------
    env = cuda_env(info[2] if len(info) > 2 else "")
    print("Installing PyTorch 2.4 + TRELLIS dependencies (15-25 minutes)...", flush=True)
    pip(*TORCH)
    pip(*XFORMERS)
    pip(*BASIC)

    if not (TRELLIS_DIR / "trellis").exists():
        shutil.rmtree(TRELLIS_DIR, ignore_errors=True)
        run(["git", "clone", "--depth", "1", "https://github.com/microsoft/TRELLIS.git", TRELLIS_DIR])

    EXT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building GPU extensions (nvdiffrast, gaussian rasterizer)...", flush=True)
    nvd = EXT_DIR / "nvdiffrast"
    if not nvd.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/NVlabs/nvdiffrast.git", nvd])
    run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", nvd], env=env)

    mip = EXT_DIR / "mip-splatting"
    if not mip.exists():
        run(["git", "clone", "--depth", "1", "--recurse-submodules",
             "https://github.com/autonomousvision/mip-splatting.git", mip])
    run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation",
         mip / "submodules" / "diff-gaussian-rasterization"], env=env)

    # Quick import test so a broken install fails here, not after the model download.
    run([sys.executable, "-c",
         "import torch, xformers, spconv, nvdiffrast.torch, diff_gaussian_rasterization, utils3d;"
         "assert torch.cuda.is_available(), 'CUDA not available';"
         "print('torch', torch.__version__, 'cuda ok')"])
    # Import TRELLIS itself too, so a missing library shows up now and not after the long install.
    test_env = dict(os.environ, ATTN_BACKEND="xformers", SPCONV_ALGO="native")
    run([sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(TRELLIS_DIR)!r}); "
         "from trellis.pipelines import TrellisImageTo3DPipeline; "
         "from trellis.utils import postprocessing_utils; print('trellis import ok')"],
        env=test_env)
    DONE_FLAG.write_text("ok")
    print("Install finished.", flush=True)


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
    print(f"Done: {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


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
        return
    if not DONE_FLAG.exists():
        sys.exit("Run 'python gpu/trellis_gpu.py install' first.")
    if not in_venv():  # hop into the Python 3.10 environment
        args = [VENV_PY, "-u", HERE, "generate", str(Path(a.image).resolve()),
                "-o", str(Path(a.output).resolve()), "--seed", a.seed,
                "--texture", a.texture, "--simplify", a.simplify]
        sys.exit(subprocess.run([str(x) for x in args]).returncode)
    generate(a.image, a.output, a.seed, a.texture, a.simplify)


if __name__ == "__main__":
    main()
