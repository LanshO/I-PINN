from __future__ import annotations

import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.tensorboard import SummaryWriter

from constraints import compute_image_warp_loss, compute_raw_losses
from io_utils import save_case_metadata, save_csv_fields, save_identified_parameters


STAGE2_REFERENCE_LAMBDAS = {
    "loss_displacement_bc": 1e7,
    "loss_stress_up_down_bc": 1e7,
    "loss_stress_left_right_bc": 1e7,
    "loss_circular_hole_stress_bc": 1e8,
    "loss_inner_balance": 1e12,
    "loss_image_warp": 1e4,
    "loss_energy_work": 1.0,
}


class IPINNTrainer:
    """Framework core for image-constrained and physics-constrained inverse PINNs.

    This trainer owns the stage scheduling, optimizers, checkpoint logic, and
    static loss-weight estimation. It does not define case geometry or target
    values itself; those are injected through the case object and its registered
    constraint collection.
    """

    def __init__(self, case, model, material_parameters, save_path):
        self.case = case
        self.model = model
        self.material = material_parameters
        self.device = case.device
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        self.criterion = torch.nn.MSELoss()
        self.case_tensors = self.case.build_case_tensors()
        self.constraint_collection = self.case.build_constraint_data(self.case_tensors)

        # Stage 1 uses dense image matching only. Stage 2 combines image and
        # physics constraints through the unified raw-loss dictionary.
        self.stage1_epochs = 3000
        # This hook is kept from the validated single-hole workflow so that
        # stage-2 physics terms can be rescaled later without changing the
        # generic loss-registration interface.
        self.stage2_ramp_factors = {name: 1.0 for name in STAGE2_REFERENCE_LAMBDAS}

        self.best_loss = float("inf")
        self.iter = 0
        self.stage2_target_ratios = None
        self.stage2_estimated_lambdas = None
        self.stage2_initial_raw_losses = None
        self.stage_mode = "stage1"

        self._init_optimizers()
        self.writer = SummaryWriter(os.path.join(save_path, "logs"))

    @property
    def E(self):
        return self.material.E

    @property
    def v(self):
        return self.material.v

    def _all_train_parameters(self):
        model_parameters = list(self.model.parameters())
        material_parameters = list(self.material.parameters())
        return model_parameters + material_parameters

    def _init_optimizers(self):
        all_train_parameters = self._all_train_parameters()
        material_parameters = list(self.material.parameters())
        self.stage2_optimizer = torch.optim.LBFGS(
            all_train_parameters,
            lr=0.1,
            max_iter=50000,
            tolerance_grad=1e-12,
            line_search_fn="strong_wolfe",
        )
        self.LBFGS_2 = None
        if material_parameters:
            self.LBFGS_2 = torch.optim.LBFGS(
                material_parameters,
                lr=0.1,
                max_iter=500,
                tolerance_grad=1e-12,
                line_search_fn="strong_wolfe",
            )
        # Adam is used only in stage 1 to pre-align the deformed image.
        self.adam = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-6)

    def save_checkpoint(self, filename, is_best=False):
        """Save the framework state, including model and material modules."""

        state = {
            "model_state_dict": self.model.state_dict(),
            "material_state_dict": self.material.state_dict(),
            "stage2_optimizer_state_dict": self.stage2_optimizer.state_dict(),
            "adam_state_dict": self.adam.state_dict(),
            "LBFGS_2_state_dict": self.LBFGS_2.state_dict() if self.LBFGS_2 is not None else None,
            "iter": self.iter,
            "best_loss": self.best_loss,
            "stage2_target_ratios": self.stage2_target_ratios,
            "stage2_estimated_lambdas": self.stage2_estimated_lambdas,
            "stage2_initial_raw_losses": self.stage2_initial_raw_losses,
        }
        torch.save(state, os.path.join(self.save_path, filename))
        if is_best:
            torch.save(state, os.path.join(self.save_path, "best_model.pth"))

    def load_checkpoint(self, filename):
        """Load the framework state and resume from the stored iteration."""

        checkpoint = torch.load(os.path.join(self.save_path, filename), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "material_state_dict" in checkpoint:
            self.material.load_state_dict(checkpoint["material_state_dict"])
        optimizer_state = checkpoint.get("stage2_optimizer_state_dict", checkpoint.get("optimizer_state_dict"))
        if optimizer_state is not None:
            self.stage2_optimizer.load_state_dict(optimizer_state)
        self.adam.load_state_dict(checkpoint["adam_state_dict"])
        if self.LBFGS_2 is not None and checkpoint.get("LBFGS_2_state_dict") is not None:
            self.LBFGS_2.load_state_dict(checkpoint["LBFGS_2_state_dict"])
        self.iter = checkpoint["iter"]
        self.best_loss = checkpoint["best_loss"]
        self.stage2_target_ratios = checkpoint.get("stage2_target_ratios")
        self.stage2_estimated_lambdas = checkpoint.get("stage2_estimated_lambdas")
        self.stage2_initial_raw_losses = checkpoint.get("stage2_initial_raw_losses")
        print(f"Loaded checkpoint from {filename}, resuming from iteration {self.iter}")

    def _derive_fields(self):
        """Forward pass plus automatic differentiation for displacement, strain, and stress.

        The returned tensors are reused by multiple generic loss formulas:
        - displacement enters image and displacement-BC losses,
        - strains and stresses enter constitutive, energy, and export logic,
        - stress gradients enter the equilibrium loss.
        """

        output_fullfield_dis = self.model(self.case_tensors.all_input_coordinates)
        u = output_fullfield_dis[:, 0]
        v = output_fullfield_dis[:, 1]

        du_dX = torch.autograd.grad(
            outputs=u,
            inputs=self.case_tensors.all_input_coordinates,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
        )[0]
        dv_dX = torch.autograd.grad(
            outputs=v,
            inputs=self.case_tensors.all_input_coordinates,
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=True,
        )[0]

        du_dx = du_dX[:, 0] / self.case.scale_W
        du_dy = du_dX[:, 1] / self.case.scale_H
        dv_dx = dv_dX[:, 0] / self.case.scale_W
        dv_dy = dv_dX[:, 1] / self.case.scale_H

        # Plane-stress constitutive response used by every physics loss.
        sigma_xx, sigma_xy, sigma_yy = self.material.stress_components(du_dx, dv_dy, du_dy, dv_dx)

        sigma_xx_dX = torch.autograd.grad(
            outputs=sigma_xx,
            inputs=self.case_tensors.all_input_coordinates,
            grad_outputs=torch.ones_like(sigma_xx),
            retain_graph=True,
            create_graph=True,
        )[0]
        sigma_xy_dX = torch.autograd.grad(
            outputs=sigma_xy,
            inputs=self.case_tensors.all_input_coordinates,
            grad_outputs=torch.ones_like(sigma_xy),
            retain_graph=True,
            create_graph=True,
        )[0]
        sigma_yy_dX = torch.autograd.grad(
            outputs=sigma_yy,
            inputs=self.case_tensors.all_input_coordinates,
            grad_outputs=torch.ones_like(sigma_yy),
            retain_graph=True,
            create_graph=True,
        )[0]

        return {
            "displacement": output_fullfield_dis,
            # Strain tuple ordering follows (epsilon_xx, du/dy, dv/dx, epsilon_yy).
            # The shear-related terms are kept separately so energy and export
            # code can build either tensor shear strain or engineering shear strain.
            "strains": (du_dx, du_dy, dv_dx, dv_dy),
            "stress": (sigma_xx, sigma_xy, sigma_yy),
            # Stress-gradient ordering is chosen to make div(sigma) assembly
            # explicit inside the equilibrium loss.
            "stress_gradients": (
                sigma_xx_dX[:, 0] / self.case.scale_W,
                sigma_xx_dX[:, 1] / self.case.scale_H,
                sigma_xy_dX[:, 0] / self.case.scale_W,
                sigma_xy_dX[:, 1] / self.case.scale_H,
                sigma_yy_dX[:, 0] / self.case.scale_W,
                sigma_yy_dX[:, 1] / self.case.scale_H,
            ),
        }

    def compute_stage2_raw_losses(self, result):
        """Ask the generic constraint module to evaluate all enabled loss specs."""

        return compute_raw_losses(self.case, self.constraint_collection, result, self.criterion)

    def estimate_target_based_weights(self, raw_losses, target_ratios):
        """Estimate the manuscript-style static weights from raw-loss targets.

        Each target ratio t_k represents a desired relative reduction level of
        the raw stage-2 loss L_k^0. The corresponding weight is:

            lambda_k = 1 / (t_k * L_k^0)
        """

        eps = 1e-12
        estimated = {}
        for name, loss_value in raw_losses.items():
            scalar_loss = max(float(loss_value.detach().item()), eps)
            estimated[name] = 1.0 / (target_ratios[name] * scalar_loss)
        return estimated

    def ensure_stage2_weight_configuration(self, raw_losses):
        """Create stage-2 static weights once and keep them fixed during training.

        The default target ratios are back-calculated from the validated
        single-hole reference lambdas so that the resulting weights stay close to
        the original scale while following the target-based static strategy.
        """

        if self.stage2_target_ratios is None or self.stage2_estimated_lambdas is None:
            eps = 1e-12
            self.stage2_initial_raw_losses = {
                name: float(loss_value.detach().item()) for name, loss_value in raw_losses.items()
            }
            self.stage2_target_ratios = {
                name: 1.0 / (STAGE2_REFERENCE_LAMBDAS[name] * max(self.stage2_initial_raw_losses[name], eps))
                for name in raw_losses
            }
            self.stage2_estimated_lambdas = self.estimate_target_based_weights(raw_losses, self.stage2_target_ratios)

            print("Initialized target-based stage-2 weights:")
            for name in raw_losses:
                print(
                    f"  {name}: lambda={self.stage2_estimated_lambdas[name]:.6e}, "
                    f"target_ratio={self.stage2_target_ratios[name]:.6e}, "
                    f"raw_loss0={self.stage2_initial_raw_losses[name]:.6e}"
                )
        return self.stage2_estimated_lambdas

    def visualize_fields(self, tag_prefix="train"):
        """Export predicted fields and write TensorBoard visualizations."""

        X = self.case_tensors.norm_img_coordinates.requires_grad_(True)
        pred = self.model(X)

        u = pred[:, 0].detach().cpu().numpy().reshape(self.case.W - 2 * self.case.edge_w, self.case.H - 2 * self.case.edge_h)
        v = pred[:, 1].detach().cpu().numpy().reshape(self.case.W - 2 * self.case.edge_w, self.case.H - 2 * self.case.edge_h)

        u_tensor = pred[:, 0:1]
        v_tensor = pred[:, 1:2]

        du_dx = torch.autograd.grad(u_tensor, X, grad_outputs=torch.ones_like(u_tensor), retain_graph=True, create_graph=True)[0][:, 0:1] / self.case.scale_W
        dv_dy = torch.autograd.grad(v_tensor, X, grad_outputs=torch.ones_like(v_tensor), retain_graph=True, create_graph=True)[0][:, 1:2] / self.case.scale_H
        du_dy = torch.autograd.grad(u_tensor, X, grad_outputs=torch.ones_like(u_tensor), retain_graph=True, create_graph=True)[0][:, 1:2] / self.case.scale_H
        dv_dx = torch.autograd.grad(v_tensor, X, grad_outputs=torch.ones_like(v_tensor), retain_graph=True, create_graph=True)[0][:, 0:1] / self.case.scale_W

        sigma_xx, sigma_xy, sigma_yy = self.material.stress_components(du_dx, dv_dy, du_dy, dv_dx)

        sigma_xx = sigma_xx.detach().cpu().numpy().reshape(self.case.W - 2 * self.case.edge_w, self.case.H - 2 * self.case.edge_h)
        sigma_yy = sigma_yy.detach().cpu().numpy().reshape(self.case.W - 2 * self.case.edge_w, self.case.H - 2 * self.case.edge_h)
        sigma_xy = sigma_xy.detach().cpu().numpy().reshape(self.case.W - 2 * self.case.edge_w, self.case.H - 2 * self.case.edge_h)

        strain_xx = du_dx.detach().cpu().numpy().reshape(self.case.W - 2 * self.case.edge_w, self.case.H - 2 * self.case.edge_h)
        strain_yy = dv_dy.detach().cpu().numpy().reshape(self.case.W - 2 * self.case.edge_w, self.case.H - 2 * self.case.edge_h)
        strain_xy = (0.5 * (du_dy + dv_dx)).detach().cpu().numpy().reshape(
            self.case.W - 2 * self.case.edge_w, self.case.H - 2 * self.case.edge_h
        )

        def add_figure_to_tb(tag, data):
            plt.figure(figsize=(8, 4))
            sns.heatmap(data, cmap="jet", cbar=True)
            plt.title(tag)
            plt.axis("off")
            self.writer.add_figure(tag_prefix + "/" + tag, plt.gcf(), self.iter)
            plt.close()

        for name, data in [
            ("Displacement_u", u),
            ("Displacement_v", v),
            ("Stress_xx", sigma_xx),
            ("Stress_yy", sigma_yy),
            ("Stress_xy", sigma_xy),
            ("Strain_xx", strain_xx),
            ("Strain_yy", strain_yy),
            ("Strain_xy", strain_xy),
        ]:
            add_figure_to_tb(name, data)

        field_map = {
            "displacement_u.csv": u,
            "displacement_v.csv": v,
            "stress_xx.csv": sigma_xx,
            "stress_yy.csv": sigma_yy,
            "stress_xy.csv": sigma_xy,
            "strain_xx.csv": strain_xx,
            "strain_yy.csv": strain_yy,
            "strain_xy.csv": strain_xy,
        }

        if tag_prefix == "eval":
            save_csv_fields(self.save_path, field_map)
            save_identified_parameters(
                self.save_path,
                self.E.item(),
                self.v.item(),
                source=self.material.parameter_source,
            )
            save_case_metadata(self.save_path, self.case.metadata())
        else:
            save_csv_fields(self.save_path, {f"Img_MSE_{name}": array for name, array in field_map.items()})

    def _compute_total_loss(self):
        if self.stage_mode == "stage1":
            self.adam.zero_grad()
            self.stage2_optimizer.zero_grad()

            output_fullfield_dis = self.model(self.case_tensors.all_input_coordinates)
            image_spec = self.constraint_collection.specs["image_warp"]
            raw_image_warp = compute_image_warp_loss(
                self.case, image_spec, output_fullfield_dis, self.criterion
            )
            total_loss = raw_image_warp * STAGE2_REFERENCE_LAMBDAS["loss_image_warp"]
            self.writer.add_scalar("Loss/Image_Warp", total_loss, self.iter)
            return total_loss, total_loss

        result = self._derive_fields()
        raw_loss_dict = self.compute_stage2_raw_losses(result)
        estimated_lambdas = self.ensure_stage2_weight_configuration(raw_loss_dict)

        magnitude_loss_dict = {
            name: raw_loss_dict[name] * estimated_lambdas[name] * self.stage2_ramp_factors[name]
            for name in raw_loss_dict
        }
        total_loss = sum(magnitude_loss_dict.values())

        self.adam.zero_grad()
        if self.LBFGS_2 is not None:
            self.LBFGS_2.zero_grad()
        self.stage2_optimizer.zero_grad()

        for scalar_name, value in magnitude_loss_dict.items():
            self.writer.add_scalar(f"Loss/{scalar_name}", value, self.iter)
        self.writer.add_scalar("Loss/Poisson_Ratio", self.v, self.iter)
        self.writer.add_scalar("Loss/Young_modulus", self.E, self.iter)

        if self.material.is_trainable and self.iter % 100 == 0:
            print("Young_modulus", self.E.item())
            print("Poisson_Ratio", self.v.item())

        return total_loss, total_loss

    def loss_func(self):
        total_loss, writer_total = self._compute_total_loss()
        self.writer.add_scalar("Loss/Total", writer_total, self.iter)

        total_loss.backward()
        if self.iter % 100 == 0:
            print("Iteration_num", self.iter)
            print("Sum_loss", total_loss.item())

        if self.iter % 1000 == 0:
            self.save_checkpoint("checkpoint.pth")

        if total_loss.item() < self.best_loss:
            self.best_loss = total_loss.item()
            self.save_checkpoint("best_model.pth", is_best=True)

        return total_loss

    def train(self, resume=False, checkpoint_file=None):
        """Run Adam image pre-training followed by full LBFGS refinement."""

        if resume and checkpoint_file:
            self.load_checkpoint(checkpoint_file)
        else:
            self.iter = 0

        self.model.train()

        def closure():
            self.stage_mode = "stage1"
            self.iter += 1
            return self.loss_func()

        def stage2_closure():
            self.stage_mode = "stage2"
            self.iter += 1
            return self.loss_func()

        for i in range(self.iter, self.stage1_epochs):
            self.adam.step(closure)
            if (i + 1) % 1000 == 0:
                self.save_checkpoint("checkpoint.pth")
                print(f"Adam iteration {i + 1}/{self.stage1_epochs}")

        self.visualize_fields("train")
        self.stage2_optimizer.step(stage2_closure)

        print("Material_source", self.material.parameter_source)
        print("Young_modulus", self.E.item())
        print("Poisson_Ratio", self.v.item())
        self.save_checkpoint("checkpoint.pth")
        self.writer.close()

    def evaluate(self):
        """Export the final predicted fields using the evaluation naming convention."""

        self.model.eval()
        self.visualize_fields("eval")
        print(f"Evaluation results saved to {self.save_path}")
