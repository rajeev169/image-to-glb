# image-to-glb

Turns one image into a textured 3D model (GLB) with Microsoft TRELLIS.
Pick where it runs from a dropdown in GitHub Actions:

| Choice | Where it runs | Speed | Limit | Setup |
|---|---|---|---|---|
| **huggingface** | Free TRELLIS Space | 3–5 min | Small daily GPU quota | `HF_TOKEN` secret (recommended) |
| **kaggle** | Your Kaggle account, free T4 GPU | 25–35 min | ~30 GPU hours/week | `KAGGLE_API_TOKEN` secret |
| **colab** | Google Colab, free T4 GPU | 20–30 min | Colab daily limit | Nothing – opens a link |

## Run
Actions tab → **Image to GLB** → Run workflow → choose **backend** and the **image path** (e.g. `rider_white_bg.png`) → Run.
When it finishes, download the **model-glb** artifact at the bottom of the run page.
For **colab**, open the run's Summary and tap the "Open in Colab" link.

## One-time setup (repo → Settings → Secrets and variables → Actions → New repository secret)
- `HF_TOKEN` – huggingface.co → Settings → Access Tokens → New token (Read). Without it, GitHub machines share one tiny anonymous quota.
- `KAGGLE_API_TOKEN` – kaggle.com → Settings → API → **Generate New Token**. Your Kaggle account must be **phone-verified** (needed for GPU + internet).

## Tips for good results
- Whole object visible, plain or transparent background, three-quarter angle.
- Clean edges matter: dark specks around the object can turn into floating bits.
- `input_used.png` in the Kaggle artifact shows the cut-out image TRELLIS actually used.
- Rider and bike come out as one static mesh (no skeleton/animation).
- Material is set to non-metallic so it isn't black in viewers/Unity (Unity needs the glTFast package for .glb).

## Files
- `trellis_image_to_glb.py` – Hugging Face path
- `gpu/trellis_gpu.py` – installs and runs TRELLIS on a free GPU (used by Kaggle and Colab)
- `tools/kaggle_run.py` – starts the Kaggle job from GitHub and downloads the GLB
- `colab/Image_to_GLB.ipynb` – Colab notebook
- `glb_utils.py` – GLB checks and material fix
