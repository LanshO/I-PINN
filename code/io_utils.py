import json
import os

import cv2
import numpy as np
import torch


def load_grayscale_image(path, device):
    """Load a grayscale speckle image and normalize it as in the research code."""

    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    img = (img - np.mean(img)) / np.max(np.abs(img - np.mean(img)))
    return torch.tensor(img).float().to(device)


def save_csv_fields(output_dir, field_map):
    """Save predicted fields using the filenames expected by the plotting scripts."""

    for filename, array in field_map.items():
        np.savetxt(os.path.join(output_dir, filename), array, delimiter=",")


def save_identified_parameters(output_dir, youngs_modulus, poisson_ratio, source="fixed"):
    """Persist the material parameters and record whether they were fixed or inferred."""

    payload = {
        "E": float(youngs_modulus),
        "nu": float(poisson_ratio),
        "source": source,
    }
    with open(os.path.join(output_dir, "identified_parameters.json"), "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def save_case_metadata(output_dir, metadata):
    """Persist case metadata so evaluation and plotting stay case driven."""

    with open(os.path.join(output_dir, "case_metadata.json"), "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)
