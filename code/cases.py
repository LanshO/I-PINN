from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np
import torch

from constraints import ConstraintCollection, ConstraintSpec
from io_utils import load_grayscale_image


@dataclass
class CaseTensors:
    """Tensors and index windows required by the I-PINN trainer for one case.

    Attributes:
        norm_img_coordinates: Dense normalized coordinates used to export the
            full predicted displacement/strain/stress fields for visualization.
        inner_coordinates: Valid image pixel coordinates outside the hole. These
            are used by the image-warp loss because warping is performed in
            pixel space rather than normalized coordinate space.
        norm_inner_coordinates: Normalized inner-domain collocation points used
            by the equilibrium loss and by automatic differentiation.
        norm_bc1/norm_bc2: Normalized coordinates on the left/right loaded
            boundaries.
        norm_bc3/norm_bc4: Normalized coordinates on the top/bottom free
            boundaries.
        norm_bc5: Normalized coordinates on the circular hole boundary.
        energy_sample_points: Normalized Monte Carlo points used by the energy
            consistency constraint.
        all_input_coordinates: Concatenated normalized coordinates passed to the
            network so that all physics derivatives can be computed in one graph.
        img_coordinates: Pixel-space coordinates used only by the image-warp
            loss.
        slices: Index windows locating each point subset inside
            all_input_coordinates.
        circle_x/circle_y: Physical x/y coordinates of the sampled hole
            boundary points, used to build outward normals.
        domain_area: Physical area of the perforated plate after subtracting
            the circular hole, used by the energy loss.
    """

    norm_img_coordinates: torch.Tensor
    inner_coordinates: torch.Tensor
    norm_inner_coordinates: torch.Tensor
    norm_bc1: torch.Tensor
    norm_bc2: torch.Tensor
    norm_bc3: torch.Tensor
    norm_bc4: torch.Tensor
    norm_bc5: torch.Tensor
    energy_sample_points: torch.Tensor
    all_input_coordinates: torch.Tensor
    img_coordinates: torch.Tensor
    slices: dict
    circle_x: torch.Tensor
    circle_y: torch.Tensor
    domain_area: float


class BaseCase:
    """Base interface for a geometry/loading case used by the I-PINN framework."""

    case_name = "base_case"

    def __init__(self, case_dir, device):
        self.case_dir = os.path.abspath(case_dir)
        self.device = device
        os.makedirs(self.case_dir, exist_ok=True)

    def build_case_tensors(self) -> CaseTensors:
        raise NotImplementedError

    def metadata(self) -> dict:
        raise NotImplementedError

    def build_constraint_data(self, case_tensors: CaseTensors) -> ConstraintCollection:
        raise NotImplementedError


class SingleHoleCase(BaseCase):
    """Single-hole benchmark passed into the generic I-PINN training framework."""

    case_name = "single_hole"

    def __init__(self, case_dir, device):
        super().__init__(case_dir, device)

        # Input images are grayscale speckle fields normalized to the same range
        # used by the validated research script.
        self.re_img = load_grayscale_image(os.path.join(self.case_dir, "re001.bmp"), self.device)
        self.tar_img = load_grayscale_image(os.path.join(self.case_dir, "tar001.bmp"), self.device)

        self.H = int(self.re_img.shape[0])
        self.W = int(self.re_img.shape[1])

        # Number of pixels cropped from the left and right sides when matching
        # the FEM field of view to the training/export field of view.
        self.edge_w = 20
        # Number of pixels cropped from the top and bottom sides for the same purpose.
        self.edge_h = 20
        # Additional inward safety margin used when sampling boundary points and
        # Monte Carlo energy points. This is distinct from edge_w/edge_h because
        # it controls collocation-point placement rather than final field cropping.
        self.edge = 10
        # Circular hole radius in pixels. It is used to remove hole-interior
        # points, generate hole-boundary points, and define the local error ring.
        self.radius = 40.0
        # Hole center in the physical coordinate system used by the case.
        # The packaged single-hole benchmark is centered at the origin.
        self.hole_center = (0.0, 0.0)

        # Prescribed horizontal displacement magnitude on the left/right
        # displacement boundary when the displacement BC is used.
        self.bc_dis = 10.0
        # Applied normal traction magnitude on the left/right loaded edges.
        self.bc_stress = 0.04
        # Reference scale used when interpreting exported stress magnitude.
        # Some historical exports store stresses in normalized GPa-scale values.
        self.stress_scale = 10**9

        # Horizontal pixel-to-normalized-coordinate scale. A physical x
        # coordinate is divided by this value before entering the network.
        self.scale_W = self.W // 2
        # Vertical pixel-to-normalized-coordinate scale used in the same way.
        self.scale_H = self.H // 2
        # Auxiliary factor used when composing some y-related residual terms in
        # the loss. It is part of the loss scaling, not a geometry parameter.
        self.scale_y = 1.0

        # Number of Monte Carlo points used to approximate the strain-energy
        # integral in the energy consistency loss.
        self.num_energy_sample_points = 20000

        # Reference material parameters used by the packaged FEM benchmark.
        self.reference_youngs_modulus = 3.0
        self.reference_poissons_ratio = 0.32

    def _build_axis_coordinates(self):
        """Return physical x/y pixel axes centered at the plate middle."""

        x = torch.arange(-self.W // 2 + 1, self.W // 2 + 1, 1, dtype=torch.float32, device=self.device)
        y = torch.arange(-self.H // 2 + 1, self.H // 2 + 1, 1, dtype=torch.float32, device=self.device)
        return x, y

    def _generate_energy_sample_points(self):
        """Sample normalized Monte Carlo points in the perforated plate."""

        points_collected = []
        num_generated = 0

        box_area = (self.W - 2 * self.edge) * (self.H - 2 * self.edge)
        hole_area = np.pi * self.radius**2
        domain_area = box_area - hole_area
        acceptance_ratio = domain_area / box_area
        num_to_generate_initially = int(self.num_energy_sample_points / acceptance_ratio * 1.1)

        while num_generated < self.num_energy_sample_points:
            x_box = (torch.rand(num_to_generate_initially, 1, device=self.device) - 0.5) * (self.W - 2 * self.edge)
            y_box = (torch.rand(num_to_generate_initially, 1, device=self.device) - 0.5) * (self.H - 2 * self.edge)
            candidate_points = torch.hstack([x_box, y_box])

            is_outside_hole = candidate_points[:, 0] ** 2 + candidate_points[:, 1] ** 2 > self.radius**2
            valid_physical_points = candidate_points[is_outside_hole]

            valid_points_normalized = torch.empty_like(valid_physical_points)
            valid_points_normalized[:, 0] = valid_physical_points[:, 0] / self.scale_W
            valid_points_normalized[:, 1] = valid_physical_points[:, 1] / self.scale_H
            points_collected.append(valid_points_normalized)

            num_generated = sum(p.shape[0] for p in points_collected)

        final_points = torch.cat(points_collected, dim=0)[: self.num_energy_sample_points, :]
        return final_points, domain_area

    def build_case_tensors(self) -> CaseTensors:
        x, y = self._build_axis_coordinates()

        # Dense normalized field coordinates used when exporting full predicted
        # fields after training. These cover the cropped field of view.
        norm_img_coordinates = torch.stack(
            torch.meshgrid(
                x[self.edge_w : -self.edge_w] / self.scale_W,
                y[self.edge_h : -self.edge_h] / self.scale_H,
            )
        ).reshape(2, -1).T

        # Candidate inner-domain points are sampled on a strided grid and then
        # filtered so that only pixels outside the hole remain.
        inner_temp = torch.stack(
            torch.meshgrid(
                x[self.edge_w : -self.edge_w : 2],
                y[self.edge_h : -self.edge_h : 2],
            )
        ).reshape(2, -1).T
        inner_physical = torch.cat(
            [p.unsqueeze(0) for p in inner_temp if (p[0] ** 2 + p[1] ** 2) > self.radius * self.radius]
        )

        # Pixel-space inner coordinates are used by image warping because the
        # interpolation is performed directly in image pixel coordinates.
        inner_coordinates = torch.ones_like(inner_physical)
        inner_coordinates[:, 0] = inner_physical[:, 0] + self.W // 2
        inner_coordinates[:, 1] = inner_physical[:, 1] + self.H // 2

        # The same inner points are also stored in normalized coordinates for
        # PDE-based losses and automatic differentiation.
        norm_inner_coordinates = torch.ones_like(inner_physical)
        norm_inner_coordinates[:, 0] = inner_physical[:, 0] / self.scale_W
        norm_inner_coordinates[:, 1] = inner_physical[:, 1] / self.scale_H

        # Boundary point sets are stored in normalized coordinates because the
        # network and automatic differentiation operate in normalized space.
        # bc1/bc2 correspond to the left/right loaded boundaries.
        norm_bc1 = torch.stack(torch.meshgrid(x[10] / self.scale_W, y[10:-10] / self.scale_H)).reshape(2, -1).T
        norm_bc2 = torch.stack(torch.meshgrid(x[-11] / self.scale_W, y[10:-10] / self.scale_H)).reshape(2, -1).T
        # bc3/bc4 correspond to the top/bottom traction-free boundaries.
        norm_bc3 = torch.stack(torch.meshgrid(x[10:-10] / self.scale_W, y[10] / self.scale_H)).reshape(2, -1).T
        norm_bc4 = torch.stack(torch.meshgrid(x[10:-10] / self.scale_W, y[-11] / self.scale_H)).reshape(2, -1).T

        # Hole boundary points are sampled uniformly in angle and later used by
        # the traction-free hole-boundary constraint.
        num_circle_points = 7200
        circle_x = torch.tensor(
            [self.radius * math.cos(i * 2 * math.pi / num_circle_points) for i in range(num_circle_points)],
            dtype=torch.float32,
            device=self.device,
        )
        circle_y = torch.tensor(
            [self.radius * math.sin(i * 2 * math.pi / num_circle_points) for i in range(num_circle_points)],
            dtype=torch.float32,
            device=self.device,
        )
        norm_bc5 = torch.stack([circle_x / self.scale_W, circle_y / self.scale_H]).reshape(2, -1).T

        # Monte Carlo points for the energy consistency loss.
        energy_sample_points, domain_area = self._generate_energy_sample_points()

        # A single concatenated coordinate tensor is used so the framework can
        # perform one forward pass and one automatic-differentiation graph build
        # for all physics-based losses.
        all_input_coordinates = torch.cat(
            [
                norm_inner_coordinates,
                norm_bc1,
                norm_bc2,
                norm_bc3,
                norm_bc4,
                norm_bc5,
                energy_sample_points,
            ],
            dim=0,
        ).requires_grad_(True)

        lengths = [
            inner_coordinates.shape[0],
            norm_bc1.shape[0],
            norm_bc2.shape[0],
            norm_bc3.shape[0],
            norm_bc4.shape[0],
            norm_bc5.shape[0],
            energy_sample_points.shape[0],
        ]

        # These slices map each logical point subset back into the concatenated
        # tensor so that loss functions can recover the relevant coordinates.
        start = 0
        slices = {}
        for length, name in zip(lengths, ["inner", "bc1", "bc2", "bc3", "bc4", "bc5", "energy"]):
            end = start + length
            slices[name] = slice(start, end)
            start = end

        return CaseTensors(
            norm_img_coordinates=norm_img_coordinates,
            inner_coordinates=inner_coordinates,
            norm_inner_coordinates=norm_inner_coordinates,
            norm_bc1=norm_bc1,
            norm_bc2=norm_bc2,
            norm_bc3=norm_bc3,
            norm_bc4=norm_bc4,
            norm_bc5=norm_bc5,
            energy_sample_points=energy_sample_points,
            all_input_coordinates=all_input_coordinates,
            img_coordinates=inner_coordinates,
            slices=slices,
            circle_x=circle_x,
            circle_y=circle_y,
            domain_area=domain_area,
        )

    def build_constraint_data(self, case_tensors: CaseTensors) -> ConstraintCollection:
        """Register all constraints required by the single-hole benchmark.

        The framework-level loss formulas are generic. This case only supplies
        which points are used, what the target values are, and which weight key
        each constraint belongs to.
        """

        bc1_len = case_tensors.norm_bc1.shape[0]
        bc2_len = case_tensors.norm_bc2.shape[0]
        hole_len = case_tensors.norm_bc5.shape[0]
        inner_len = case_tensors.inner_coordinates.shape[0]

        left_right_slice = slice(case_tensors.slices["bc1"].start, case_tensors.slices["bc2"].stop)
        up_down_slice = slice(case_tensors.slices["bc3"].start, case_tensors.slices["bc4"].stop)

        # Left/right displacement targets used by the prescribed displacement
        # boundary condition.
        left_target_u = torch.zeros(bc1_len, device=self.device) - self.bc_dis
        right_target_u = torch.zeros(bc2_len, device=self.device) + self.bc_dis
        zero_v = torch.zeros(bc1_len + bc2_len, device=self.device)

        # Outward unit normals on the circular hole boundary are needed to turn
        # stresses into tractions for the hole traction-free condition.
        normals = torch.ones_like(case_tensors.norm_bc5)
        normals[:, 0] = case_tensors.circle_x / self.radius
        normals[:, 1] = case_tensors.circle_y / self.radius

        specs = {
            # Dense image constraint on all valid pixels outside the hole.
            "image_warp": ConstraintSpec(
                name="image_warp",
                weight_key="loss_image_warp",
                coordinates=case_tensors.img_coordinates,
                slices={"all": case_tensors.slices["inner"]},
            ),
            "displacement_bc": ConstraintSpec(
                name="displacement_bc",
                weight_key="loss_displacement_bc",
                # The generic framework supports prescribed-displacement loss
                # terms, but the validated single-hole inverse benchmark follows
                # the original research configuration and does not activate this
                # term during stage 2. The packaged benchmark is driven by
                # image matching, traction conditions, equilibrium, and energy.
                enabled=False,
                coordinates=torch.cat([case_tensors.norm_bc1, case_tensors.norm_bc2], dim=0),
                target_values={
                    "u": torch.cat([left_target_u, right_target_u], dim=0),
                    "v": zero_v,
                },
                slices={"all": left_right_slice},
            ),
            # Non-zero traction condition on the left/right loaded boundaries.
            "stress_bc_left_right": ConstraintSpec(
                name="stress_bc",
                weight_key="loss_stress_left_right_bc",
                coordinates=torch.cat([case_tensors.norm_bc1, case_tensors.norm_bc2], dim=0),
                target_values={
                    "sigma_xx": torch.ones(bc1_len + bc2_len, device=self.device) * self.bc_stress,
                    "sigma_xy": torch.zeros(bc1_len + bc2_len, device=self.device),
                },
                slices={"all": left_right_slice},
            ),
            # Traction-free condition on the top and bottom plate edges.
            "stress_bc_up_down": ConstraintSpec(
                name="stress_bc",
                weight_key="loss_stress_up_down_bc",
                coordinates=torch.cat([case_tensors.norm_bc3, case_tensors.norm_bc4], dim=0),
                target_values={
                    "sigma_yy": torch.zeros(case_tensors.norm_bc3.shape[0] + case_tensors.norm_bc4.shape[0], device=self.device),
                    "sigma_xy": torch.zeros(case_tensors.norm_bc3.shape[0] + case_tensors.norm_bc4.shape[0], device=self.device),
                },
                slices={"all": up_down_slice},
            ),
            # Traction-free condition on the circular hole boundary.
            "hole_traction_free": ConstraintSpec(
                name="hole_traction_free",
                weight_key="loss_circular_hole_stress_bc",
                coordinates=case_tensors.norm_bc5,
                target_values={
                    "traction_x": torch.zeros(hole_len, device=self.device),
                    "traction_y": torch.zeros(hole_len, device=self.device),
                },
                slices={"all": case_tensors.slices["bc5"]},
                extra={"normals": normals},
            ),
            # Interior collocation points for the equilibrium equations.
            "inner_balance": ConstraintSpec(
                name="inner_balance",
                weight_key="loss_inner_balance",
                coordinates=case_tensors.norm_inner_coordinates,
                target_values={
                    "balance_x": torch.zeros(inner_len, device=self.device),
                    "balance_y": torch.zeros(inner_len, device=self.device),
                },
                slices={"all": case_tensors.slices["inner"]},
            ),
            # Energy points plus left/right loaded edges for work-energy consistency.
            "energy_consistency": ConstraintSpec(
                name="energy_consistency",
                weight_key="loss_energy_work",
                coordinates=case_tensors.energy_sample_points,
                slices={
                    "energy": case_tensors.slices["energy"],
                    "left_bc": case_tensors.slices["bc1"],
                    "right_bc": case_tensors.slices["bc2"],
                },
                extra={
                    "domain_area": case_tensors.domain_area,
                    "bc_stress": self.bc_stress,
                    "loaded_edge_length": self.H - 2 * self.edge,
                },
            ),
        }

        return ConstraintCollection(specs=specs)

    def metadata(self) -> dict:
        """Metadata consumed by evaluation and visualization scripts."""

        return {
            "case_name": self.case_name,
            "image_files": {"reference": "re001.bmp", "deformed": "tar001.bmp"},
            "field_files": {
                "u": {"prediction": "displacement_u.csv", "reference": "u001.csv"},
                "v": {"prediction": "displacement_v.csv", "reference": "v001.csv"},
                "exx": {"prediction": "strain_xx.csv", "reference": "strain_xx001.csv"},
                "exy": {"prediction": "strain_xy.csv", "reference": "strain_xy001.csv"},
                "eyy": {"prediction": "strain_yy.csv", "reference": "strain_yy001.csv"},
                "sxx": {"prediction": "stress_xx.csv", "reference": "stress_xx001.csv"},
                "sxy": {"prediction": "stress_xy.csv", "reference": "stress_xy001.csv"},
                "syy": {"prediction": "stress_yy.csv", "reference": "stress_yy001.csv"},
            },
            "crop": {"top": 20, "bottom": 20, "left": 20, "right": 20},
            "hole": {"center": "image_center", "radius": self.radius},
            "local_region": {"type": "annulus", "inner_radius": self.radius, "outer_radius": 1.5 * self.radius},
            "reference_fields": {"strain_xy_kind": "engineering_shear", "stress_unit": "Pa"},
            "predicted_fields": {"strain_xy_kind": "tensor_shear", "stress_unit": "Pa_or_normalized_gpa"},
            "true_material": {"E": self.reference_youngs_modulus, "nu": self.reference_poissons_ratio},
        }

    def write_metadata(self, output_dir: str) -> None:
        with open(os.path.join(output_dir, "case_metadata.json"), "w", encoding="utf-8") as fp:
            json.dump(self.metadata(), fp, indent=2)
