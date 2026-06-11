from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIELD_CONFIG = {
    "u": ("displacement_u.csv", "u001.csv", "u", "", "u"),
    "v": ("displacement_v.csv", "v001.csv", "v", "", "v"),
    "exx": ("strain_xx.csv", "strain_xx001.csv", r"$\varepsilon_{xx}$", "", "epsilon_xx"),
    "exy": ("strain_xy.csv", "strain_xy001.csv", r"$\varepsilon_{xy}$", "", "epsilon_xy"),
    "eyy": ("strain_yy.csv", "strain_yy001.csv", r"$\varepsilon_{yy}$", "", "epsilon_yy"),
    "sxx": ("stress_xx.csv", "stress_xx001.csv", r"$\sigma_{xx}$", "MPa", "sigma_xx"),
    "sxy": ("stress_xy.csv", "stress_xy001.csv", r"$\sigma_{xy}$", "MPa", "sigma_xy"),
    "syy": ("stress_yy.csv", "stress_yy001.csv", r"$\sigma_{yy}$", "MPa", "sigma_yy"),
}

DEFAULT_METADATA = {
    "crop": {"top": 20, "bottom": 20, "left": 20, "right": 20},
    "hole": {"radius": 40.0},
    "reference_fields": {"strain_xy_kind": "engineering_shear"},
}


def load_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


def load_case_metadata(pred_dir: Path, ref_dir: Path) -> dict:
    for root in (pred_dir, ref_dir):
        path = root / "case_metadata.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return DEFAULT_METADATA


def crop_reference(arr: np.ndarray, metadata: dict) -> np.ndarray:
    """Crop FEM fields to the same field of view used by the single-hole training script."""

    crop = metadata.get("crop", DEFAULT_METADATA["crop"])
    top = crop["top"]
    bottom = crop["bottom"]
    left = crop["left"]
    right = crop["right"]
    return arr[top : arr.shape[0] - bottom, left : arr.shape[1] - right]


def convert_reference_field(field_key: str, arr: np.ndarray, metadata: dict) -> np.ndarray:
    """Convert FEM exports to the same physical quantity used by the network outputs.

    In particular, the FEM shear-strain export is engineering shear strain
    gamma_xy, while the I-PINN prediction is the tensor shear strain
    epsilon_xy = gamma_xy / 2.
    """

    if field_key == "exy" and metadata.get("reference_fields", {}).get("strain_xy_kind") == "engineering_shear":
        return arr / 2.0
    return arr


def prediction_to_display(pred: np.ndarray, ref_shape: tuple[int, int]) -> np.ndarray:
    """Convert archived prediction CSVs to the display convention used by the MATLAB figures."""

    if pred.T.shape == ref_shape:
        return pred.T
    if pred.shape == ref_shape:
        return pred
    raise ValueError(f"Prediction shape {pred.shape} is incompatible with display reference shape {ref_shape}.")


def maybe_rescale_stress_to_mpa(pred: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_abs = float(np.nanmax(np.abs(pred)))
    ref_abs = float(np.nanmax(np.abs(ref)))
    if pred_abs > 0 and ref_abs / pred_abs >= 1e8:
        pred = pred * 1e9
    return pred / 1e6, ref / 1e6


def build_hole_mask(shape: tuple[int, int], radius: float = 40.0) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices(shape)
    cy = h // 2
    cx = w // 2
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return r2 < radius**2


def apply_hole_mask(*arrays: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    masked = []
    for arr in arrays:
        out = arr.astype(float, copy=True)
        out[mask] = np.nan
        masked.append(out)
    return masked


def masked_limits(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    lo = float(np.nanmin([np.nanmin(a), np.nanmin(b)]))
    hi = float(np.nanmax([np.nanmax(a), np.nanmax(b)]))
    return lo, hi


def robust_error_limit(err: np.ndarray) -> float:
    finite = np.abs(err[np.isfinite(err)])
    if finite.size == 0:
        return 1.0
    q = float(np.quantile(finite, 0.995))
    m = float(np.max(finite))
    if q <= 0:
        return max(m, 1.0)
    return min(max(q, 1e-12), m)


def save_comparison(pred: np.ndarray, ref: np.ndarray, title_label: str, file_label: str, output: Path, unit: str = "") -> None:
    err = pred - ref
    vmin, vmax = masked_limits(pred, ref)
    eabs = robust_error_limit(err)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    cmap_field = plt.get_cmap("jet").copy()
    cmap_error = plt.get_cmap("jet").copy()
    cmap_field.set_bad(color="white")
    cmap_error.set_bad(color="white")

    panels = [
        (pred, f"I-PINN result of {title_label}", cmap_field, vmin, vmax),
        (ref, f"FEM result of {title_label}", cmap_field, vmin, vmax),
        (err, f"Difference of {title_label} (I-PINN - FEM)", cmap_error, -eabs, eabs),
    ]

    for ax, (mat, title, cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(mat, cmap=cmap, origin="upper", vmin=lo, vmax=hi, interpolation="none")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if unit:
            cbar.set_label(unit)

    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export I-PINN / FEM / error figures for the single-hole open-source package.")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--ref-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_case_metadata(args.pred_dir, args.ref_dir)
    hole_radius = float(metadata.get("hole", {}).get("radius", DEFAULT_METADATA["hole"]["radius"]))

    for key, (pred_name, ref_name, title_label, unit, file_label) in FIELD_CONFIG.items():
        ref = crop_reference(load_matrix(args.ref_dir / ref_name), metadata)
        ref = convert_reference_field(key, ref, metadata)
        pred = prediction_to_display(load_matrix(args.pred_dir / pred_name), ref.shape)
        if key.startswith("s"):
            pred, ref = maybe_rescale_stress_to_mpa(pred, ref)
        hole_mask = build_hole_mask(ref.shape, radius=hole_radius)
        pred, ref = apply_hole_mask(pred, ref, mask=hole_mask)
        save_comparison(pred, ref, title_label, file_label, args.out_dir / f"compare_{key}.png", unit)


if __name__ == "__main__":
    main()
