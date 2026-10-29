# image-to-glb

Turns an image into a textured GLB using the free TRELLIS Space on Hugging Face.

    pip install -r requirements.txt
    python trellis_image_to_glb.py input/rider.png -o output/rider.glb

- Plain background, whole object visible, three-quarter angle works best.
- Optional: `export HF_TOKEN=...` (free Hugging Face token) for a larger GPU quota.
- Output material is set to non-metallic so it isn't black in viewers/Unity.
- GitHub Actions: Actions tab > "Image to GLB" > Run workflow > download the artifact.
