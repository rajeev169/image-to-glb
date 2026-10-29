"""Small GLB helpers shared by the Hugging Face, Kaggle and Colab paths.
Only uses the Python standard library."""
import json
import struct
from pathlib import Path


def _read(path: Path):
    data = Path(path).read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise RuntimeError("result is not a GLB file")
    total = struct.unpack("<I", data[8:12])[0]
    jlen, jtype = struct.unpack("<I4s", data[12:20])
    if jtype != b"JSON":
        raise RuntimeError("GLB has no JSON chunk")
    gltf = json.loads(data[20:20 + jlen])
    rest = data[20 + jlen:total]
    return gltf, rest


def fix_glb_material(path: Path, metallic: float = 0.0, roughness: float = 0.85) -> bool:
    """TRELLIS GLBs often leave 'metallic' unset, which viewers treat as full metal (looks black).
    Set it to non-metal. Materials that already have a metallic value or texture are left alone.
    Returns True if the file was changed."""
    gltf, rest = _read(path)
    changed = False
    for mat in gltf.get("materials", []):
        pbr = mat.setdefault("pbrMetallicRoughness", {})
        if "metallicRoughnessTexture" in pbr or "metallicFactor" in pbr:
            continue
        pbr["metallicFactor"] = metallic
        pbr["roughnessFactor"] = roughness
        changed = True
    if not changed:
        return False
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    out = b"glTF" + struct.pack("<II", 2, 20 + len(js) + len(rest))
    out += struct.pack("<I4s", len(js), b"JSON") + js + rest
    Path(path).write_bytes(out)
    return True


def check_glb(path: Path) -> None:
    """Reject obviously broken results, e.g. a flat sheet instead of a 3D object."""
    gltf, _ = _read(path)
    mins, maxs = [], []
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            idx = prim.get("attributes", {}).get("POSITION")
            if idx is None:
                continue
            acc = gltf["accessors"][idx]
            if "min" in acc and "max" in acc:
                mins.append(acc["min"])
                maxs.append(acc["max"])
    if not mins:
        raise RuntimeError("GLB has no mesh")
    lo = [min(c) for c in zip(*mins)]
    hi = [max(c) for c in zip(*maxs)]
    size = sorted(abs(h - l) for l, h in zip(lo, hi))
    if size[-1] <= 0 or size[0] / size[-1] < 0.05:
        raise RuntimeError("result is a flat sheet, not a 3D object (background was probably not removed)")
