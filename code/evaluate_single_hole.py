import argparse
import csv
import json
import math
import os

import numpy as np


FIELD_FILES = {
    "u": ("displacement_u.csv", "u001.csv"),
    "v": ("displacement_v.csv", "v001.csv"),
    "exx": ("strain_xx.csv", "strain_xx001.csv"),
    "exy": ("strain_xy.csv", "strain_xy001.csv"),
    "eyy": ("strain_yy.csv", "strain_yy001.csv"),
    "sxx": ("stress_xx.csv", "stress_xx001.csv"),
    "sxy": ("stress_xy.csv", "stress_xy001.csv"),
    "syy": ("stress_yy.csv", "stress_yy001.csv"),
}

DEFAULT_METADATA = {
    "field_files": {
        key: {"prediction": pred_name, "reference": ref_name}
        for key, (pred_name, ref_name) in FIELD_FILES.items()
    },
    "crop": {"top": 20, "bottom": 20, "left": 20, "right": 20},
    "hole": {"radius": 40.0},
    "local_region": {"inner_radius": 40.0, "outer_radius": 60.0},
    "reference_fields": {"strain_xy_kind": "engineering_shear"},
    "true_material": {"E": 3.0, "nu": 0.32},
}


def maybe_rescale_predicted_stress(field_name, field_array):
    """Support both normalized-GPa exports and already-rescaled Pa exports."""

    if not field_name.startswith("s"):
        return field_array
    max_abs_value = float(np.nanmax(np.abs(field_array)))
    if max_abs_value < 1e3:
        return field_array * 1e9
    return field_array


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the single-hole open-source results against FEM data.")
    parser.add_argument(
        "--pred-dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results")),
        help="Directory containing predicted CSV fields.",
    )
    parser.add_argument(
        "--ref-dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "example_case")),
        help="Directory containing FEM reference CSV files.",
    )
    parser.add_argument(
        "--out-file",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "metrics_summary.json")),
        help="JSON file used to store the aggregated metrics.",
    )
    return parser.parse_args()


def load_csv(path):
    return np.loadtxt(path, delimiter=",")


def load_case_metadata(pred_dir, ref_dir):
    for root in (pred_dir, ref_dir):
        path = os.path.join(root, "case_metadata.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fp:
                return json.load(fp)
    return DEFAULT_METADATA


def transform_reference(field_name, arr, metadata):
    """Match the coordinate convention used by the training script outputs.

    The FEM shear-strain export uses engineering shear strain gamma_xy,
    while the network predicts the tensor shear strain epsilon_xy.
    Therefore, gamma_xy is converted to epsilon_xy = gamma_xy / 2 before
    quantitative comparison.
    """

    crop = metadata.get("crop", DEFAULT_METADATA["crop"])
    top = crop["top"]
    bottom = crop["bottom"]
    left = crop["left"]
    right = crop["right"]
    transformed = arr[top : arr.shape[0] - bottom, left : arr.shape[1] - right].T
    if field_name == "exy" and metadata.get("reference_fields", {}).get("strain_xy_kind") == "engineering_shear":
        transformed = transformed / 2.0
    return transformed


def align_prediction_to_reference(pred, ref):
    """Handle archived exports that are stored before the final transpose."""

    if pred.shape == ref.shape:
        return pred
    if pred.T.shape == ref.shape:
        return pred.T
    raise ValueError(f"Prediction shape {pred.shape} is incompatible with reference shape {ref.shape}.")


def build_masks(shape, metadata):
    x_vals = np.arange(-(shape[0] // 2), shape[0] // 2 + 1, dtype=float)
    y_vals = np.arange(-(shape[1] // 2), shape[1] // 2 + 1, dtype=float)
    xx, yy = np.meshgrid(x_vals, y_vals, indexing="ij")
    radius = float(metadata.get("hole", {}).get("radius", DEFAULT_METADATA["hole"]["radius"]))
    outer_radius = float(metadata.get("local_region", {}).get("outer_radius", 1.5 * radius))
    r = np.sqrt(xx**2 + yy**2)
    domain_mask = r >= radius
    local_mask = (r >= radius) & (r <= outer_radius)
    return domain_mask, local_mask


def masked_errors(pred, ref, mask):
    valid = mask & np.isfinite(pred) & np.isfinite(ref)
    if not np.any(valid):
        return math.nan, math.nan
    diff = pred[valid] - ref[valid]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return mae, rmse


def maybe_load_parameters(pred_dir):
    path = os.path.join(pred_dir, "identified_parameters.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def evaluate(pred_dir, ref_dir):
    metadata = load_case_metadata(pred_dir, ref_dir)
    field_files = metadata.get("field_files", DEFAULT_METADATA["field_files"])
    pred_fields = {}
    ref_fields = {}
    for field_name, mapping in field_files.items():
        pred = load_csv(os.path.join(pred_dir, mapping["prediction"]))
        pred = maybe_rescale_predicted_stress(field_name, pred)
        ref = transform_reference(field_name, load_csv(os.path.join(ref_dir, mapping["reference"])), metadata)
        pred_fields[field_name] = align_prediction_to_reference(pred, ref)
        ref_fields[field_name] = ref

    domain_mask, local_mask = build_masks(pred_fields["u"].shape, metadata)
    metrics = {
        "pred_dir": pred_dir,
        "ref_dir": ref_dir,
        "domain_point_count": int(np.count_nonzero(domain_mask)),
        "local_point_count": int(np.count_nonzero(local_mask)),
    }

    for field_name in field_files:
        full_mae, full_rmse = masked_errors(pred_fields[field_name], ref_fields[field_name], domain_mask)
        metrics[f"{field_name}_mae"] = full_mae
        metrics[f"{field_name}_rmse"] = full_rmse
        if field_name not in ("u", "v"):
            local_mae, local_rmse = masked_errors(pred_fields[field_name], ref_fields[field_name], local_mask)
            metrics[f"{field_name}_local_mae"] = local_mae
            metrics[f"{field_name}_local_rmse"] = local_rmse

    params = maybe_load_parameters(pred_dir)
    if params is not None:
        true_material = metadata.get("true_material", DEFAULT_METADATA["true_material"])
        metrics["E_pred"] = float(params["E"])
        metrics["nu_pred"] = float(params["nu"])
        metrics["E_error"] = float(params["E"]) - float(true_material["E"])
        metrics["nu_error"] = float(params["nu"]) - float(true_material["nu"])

    return metrics


def main():
    args = parse_args()
    metrics = evaluate(os.path.abspath(args.pred_dir), os.path.abspath(args.ref_dir))
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
