import torch
import torch.nn as nn

from physics import inverse_sigmoid, map_sigmoid_to_range, plane_stress_stress_components


class BaseMaterialParameters(nn.Module):
    """Common material interface used by the I-PINN framework.

    Both fixed-parameter mode and inverse-identification mode expose the same
    physical properties and constitutive response, so the trainer does not need
    to branch on material mode while assembling stresses and losses.
    """

    is_trainable = False
    parameter_source = "fixed"

    @property
    def E(self):
        raise NotImplementedError

    @property
    def nu(self):
        raise NotImplementedError

    @property
    def v(self):
        """Alias kept for compatibility with the original research script."""

        return self.nu

    def stress_components(self, du_dx, dv_dy, du_dy, dv_dx):
        """Return plane-stress components for the current material state."""

        return plane_stress_stress_components(du_dx, dv_dy, du_dy, dv_dx, self.E, self.nu)


class FixedMaterialParameters(BaseMaterialParameters):
    """Material mode used when inverse identification is disabled.

    The public open-source package defaults to this mode so that the framework
    can be run as a field-reconstruction example without optimizing E and nu.
    """

    is_trainable = False
    parameter_source = "fixed"

    def __init__(self, device, youngs_modulus, poisson_ratio):
        super().__init__()
        self.register_buffer("_E", torch.tensor(float(youngs_modulus), dtype=torch.float32, device=device))
        self.register_buffer("_nu", torch.tensor(float(poisson_ratio), dtype=torch.float32, device=device))

    @property
    def E(self):
        return self._E

    @property
    def nu(self):
        return self._nu


class TrainableMaterialParameters(BaseMaterialParameters):
    """Trainable plane-stress material parameters used in inverse mode.

    The raw scalars are optimized in an unconstrained space and then mapped to
    bounded physical intervals through a sigmoid transformation. This preserves
    stable optimization while enforcing admissible material ranges.
    """

    is_trainable = True
    parameter_source = "inferred"

    def __init__(
        self,
        device,
        youngs_init=2.0,
        poisson_init=0.2,
        youngs_range=(0.0, 5.0),
        poisson_range=(-1.0, 0.5),
    ):
        super().__init__()
        self.device = device
        self.E_min, self.E_max = youngs_range
        self.nu_min, self.nu_max = poisson_range

        # The inverse problem is optimized in raw parameter space because the
        # bounded sigmoid mapping keeps the physical values inside admissible ranges.
        e_raw_init = inverse_sigmoid(youngs_init, self.E_min, self.E_max)
        nu_raw_init = inverse_sigmoid(poisson_init, self.nu_min, self.nu_max)

        self.E_raw = nn.Parameter(torch.tensor(e_raw_init, dtype=torch.float32, device=device))
        self.nu_raw = nn.Parameter(torch.tensor(nu_raw_init, dtype=torch.float32, device=device))

    @property
    def E(self):
        """Young's modulus in the physical interval configured for this case."""

        return map_sigmoid_to_range(self.E_raw, self.E_min, self.E_max)

    @property
    def nu(self):
        """Poisson's ratio in the physical interval configured for this case."""

        return map_sigmoid_to_range(self.nu_raw, self.nu_min, self.nu_max)
