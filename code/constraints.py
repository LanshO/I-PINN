from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import torch


@dataclass
class ConstraintSpec:
    """Case-provided data for one loss item in the generic I-PINN framework.

    Each spec contains the coordinates or slices involved in the constraint, the
    target values if they are known explicitly, and any extra geometry metadata
    needed by the fixed loss formula.

    Attributes:
        name: Generic loss-formula identifier consumed by the trainer. Examples
            include image_warp, displacement_bc, stress_bc, and inner_balance.
        weight_key: Key used to look up the corresponding stage-2 weight.
        enabled: Whether the current case activates this constraint.
        coordinates: Case-provided coordinates associated with this constraint.
            For image warp they remain in pixel space; for physics losses they
            are normalized physical coordinates.
        target_values: Case-provided true values or zero targets. Examples are
            prescribed displacement, boundary traction components, or zero
            balance residuals.
        slices: Index windows locating the relevant points inside the unified
            all_input_coordinates tensor built by the case.
        extra: Auxiliary physical metadata required by the fixed formula, such
            as hole normals, integration area, or loaded-edge length.
    """

    name: str
    weight_key: str
    enabled: bool = True
    coordinates: torch.Tensor | None = None
    target_values: dict[str, torch.Tensor] = field(default_factory=dict)
    slices: dict[str, slice] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


@dataclass
class ConstraintCollection:
    """Grouped constraints registered by a case for the trainer."""

    specs: dict[str, ConstraintSpec]


def compute_displacement_bc_loss(case, spec: ConstraintSpec, displacement, criterion):
    """Prescribed displacement constraint.

    The formula is generic: predicted displacement at the registered boundary
    points is matched against case-provided target displacement values.
    In the packaged single-hole case this spec is attached to the left and
    right outer edges, where u = +/- bc_dis and v = 0.
    """

    pred_u = displacement[spec.slices["all"], 0]
    pred_v = displacement[spec.slices["all"], 1]
    return criterion(pred_u, spec.target_values["u"]) + criterion(pred_v, spec.target_values["v"])


def compute_stress_bc_loss(case, spec: ConstraintSpec, stress, criterion):
    """Boundary traction constraint for outer edges.

    The case provides the boundary slice and the scalar targets for sigma_xx,
    sigma_yy, and sigma_xy, so the trainer can reuse the same formula for
    traction-free and non-zero traction boundaries.

    In the packaged single-hole case this same formula is reused for:
    - left/right loaded edges with non-zero normal traction and zero shear,
    - top/bottom free edges with zero normal traction and zero shear.
    """

    sigma_xx = stress[0][spec.slices["all"]]
    sigma_xy = stress[1][spec.slices["all"]]
    sigma_yy = stress[2][spec.slices["all"]]

    loss = 0.0
    if "sigma_xx" in spec.target_values:
        loss = loss + criterion(sigma_xx, spec.target_values["sigma_xx"])
    if "sigma_yy" in spec.target_values:
        loss = loss + criterion(sigma_yy, spec.target_values["sigma_yy"])
    if "sigma_xy" in spec.target_values:
        loss = loss + case.scale_y * criterion(sigma_xy, spec.target_values["sigma_xy"])
    return loss


def compute_hole_traction_free_loss(case, spec: ConstraintSpec, stress, criterion):
    """Hole-boundary traction-free constraint.

    The formula is fixed, while the case provides the hole-boundary slice and
    outward normal vectors through the spec metadata.
    For the single-hole benchmark this enforces zero traction on the circular
    hole boundary rather than directly matching stress components.
    """

    sigma_xx = stress[0][spec.slices["all"]]
    sigma_yy = stress[2][spec.slices["all"]]
    sigma_xy = stress[1][spec.slices["all"]]

    normals = spec.extra["normals"]
    traction_x = sigma_xx * normals[:, 0] + sigma_xy * normals[:, 1]
    traction_y = sigma_xy * normals[:, 0] + sigma_yy * normals[:, 1]

    return criterion(traction_x, spec.target_values["traction_x"]) + case.scale_y * criterion(
        traction_y, spec.target_values["traction_y"]
    )


def compute_inner_balance_loss(case, spec: ConstraintSpec, stress_gradients, criterion):
    """Interior equilibrium constraint.

    The formula is fixed: the two divergence components of the Cauchy stress
    must vanish at the case-provided interior collocation points.
    In the packaged case these points lie in the perforated plate domain
    outside the circular hole.
    """

    inner_slice = spec.slices["all"]
    balance_x = stress_gradients[0][inner_slice] + stress_gradients[3][inner_slice]
    balance_y = stress_gradients[2][inner_slice] + stress_gradients[5][inner_slice]
    return criterion(balance_x, spec.target_values["balance_x"]) + case.scale_y * criterion(
        balance_y, spec.target_values["balance_y"]
    )


def compute_energy_loss(case, spec: ConstraintSpec, displacement, strains, stress, criterion):
    """Energy consistency constraint.

    The strain energy is integrated over the case-provided energy points and
    matched to the external work computed from the case-provided loaded edges.
    For the packaged single-hole benchmark, the external work comes from the
    prescribed normal traction on the left and right outer boundaries.
    """

    energy_slice = spec.slices["energy"]
    left_slice = spec.slices["left_bc"]
    right_slice = spec.slices["right_bc"]

    sigma_xx = stress[0][energy_slice]
    sigma_yy = stress[2][energy_slice]
    sigma_xy = stress[1][energy_slice]

    e_xx = strains[0][energy_slice]
    e_yy = strains[3][energy_slice]
    e_xy = strains[1][energy_slice] + strains[2][energy_slice]

    dis_x_left = displacement[left_slice, 0]
    dis_x_right = displacement[right_slice, 0]

    energy_density = 0.5 * (sigma_xx * e_xx + sigma_yy * e_yy + sigma_xy * e_xy)
    total_energy = torch.mean(energy_density) * spec.extra["domain_area"]
    external_work = (
        torch.mean(torch.abs(dis_x_left)) * spec.extra["bc_stress"] * spec.extra["loaded_edge_length"]
        + torch.mean(torch.abs(dis_x_right)) * spec.extra["bc_stress"] * spec.extra["loaded_edge_length"]
    )
    return criterion(total_energy, 0.5 * external_work)


def compute_image_warp_loss(case, spec: ConstraintSpec, displacement, criterion, enable_debug_plot=False):
    """Dense image-matching constraint outside the hole.

    The image-warp formula itself is fixed. The case only provides which pixel
    coordinates are valid for the current geometry and field of view.
    In the single-hole case, only pixels in the cropped field of view and
    outside the circular void participate in this loss.
    """

    img_displacement = displacement[spec.slices["all"]]

    def cubic(x):
        absx = torch.abs(x)
        absx2 = absx**2
        absx3 = absx**3
        return (1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) + (
            -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
        ) * ((1 < absx) & (absx <= 2))

    def bicubic_interpolation_2d(image, x, y):
        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()

        x_bases = torch.stack([cubic(x0 + dx - x) for dx in range(-1, 3)], dim=-1)
        y_bases = torch.stack([cubic(y0 + dy - y) for dy in range(-1, 3)], dim=-1)

        interp_val = 0
        for dx in range(-1, 3):
            for dy in range(-1, 3):
                clamped_y = torch.clamp(y0 + dy, 0, case.H - 1)
                clamped_x = torch.clamp(x0 + dx, 0, case.W - 1)
                val = image[..., clamped_y, clamped_x]
                interp_val += val * x_bases[..., dx + 1] * y_bases[..., dy + 1]
        return interp_val

    pixel_coordinates = spec.coordinates
    deform_coordinates = pixel_coordinates + img_displacement
    warp_tar_img = bicubic_interpolation_2d(case.tar_img, deform_coordinates[:, 0], deform_coordinates[:, 1])
    re_speckle_img = case.re_img[..., pixel_coordinates[:, 1].long(), pixel_coordinates[:, 0].long()]
    loss_image_warp = criterion(re_speckle_img, warp_tar_img)

    if enable_debug_plot:
        original_img = re_speckle_img.detach().cpu().numpy().reshape(case.W - 2 * case.edge_w, case.H - 2 * case.edge_h)
        warped_img = warp_tar_img.detach().cpu().numpy().reshape(case.W - 2 * case.edge_w, case.H - 2 * case.edge_h)
        difference = abs(original_img - warped_img)
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(original_img, cmap="gray")
        plt.title("Reference")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(warped_img, cmap="gray")
        plt.title("Warped")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(difference, cmap="hot")
        plt.title("Difference")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return loss_image_warp


def compute_raw_losses(case, constraint_collection: ConstraintCollection, result, criterion):
    """Compute all enabled raw losses registered by the case.

    The trainer does not decide where the constraints come from; it only asks
    for the raw loss value associated with each registered spec.
    This keeps the loss formulas generic while letting each case decide which
    boundaries, targets, and collocation sets are active.
    """

    losses = {}
    specs = constraint_collection.specs

    for name, spec in specs.items():
        if not spec.enabled:
            continue

        if spec.name == "image_warp":
            losses[spec.weight_key] = compute_image_warp_loss(case, spec, result["displacement"], criterion)
        elif spec.name == "displacement_bc":
            losses[spec.weight_key] = compute_displacement_bc_loss(case, spec, result["displacement"], criterion)
        elif spec.name == "stress_bc":
            losses[spec.weight_key] = compute_stress_bc_loss(case, spec, result["stress"], criterion)
        elif spec.name == "hole_traction_free":
            losses[spec.weight_key] = compute_hole_traction_free_loss(case, spec, result["stress"], criterion)
        elif spec.name == "inner_balance":
            losses[spec.weight_key] = compute_inner_balance_loss(case, spec, result["stress_gradients"], criterion)
        elif spec.name == "energy_consistency":
            losses[spec.weight_key] = compute_energy_loss(
                case, spec, result["displacement"], result["strains"], result["stress"], criterion
            )
        else:
            raise ValueError(f"Unsupported constraint type: {spec.name}")

    return losses
