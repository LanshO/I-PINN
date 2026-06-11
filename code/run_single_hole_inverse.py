import argparse
import os

import torch

from cases import SingleHoleCase
from framework import IPINNTrainer
from materials import FixedMaterialParameters, TrainableMaterialParameters
from model import build_model

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the open-source I-PINN framework on the packaged single-hole case.")
    parser.add_argument(
        "--case-dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "example_case")),
        help="Directory containing the case images and FEM reference data. Outputs are written to the same directory.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default="checkpoint.pth",
        help="Checkpoint filename inside case-dir used when --resume is enabled.",
    )
    parser.add_argument(
        "--activation",
        choices=("gelu", "tanh"),
        default="gelu",
        help="Hidden-layer activation used by the selected backbone.",
    )
    parser.add_argument(
        "--model-name",
        choices=("resnn", "plainnn"),
        default="resnn",
        help="Backbone model passed into the generic I-PINN framework.",
    )
    parser.add_argument(
        "--invert-material",
        action="store_true",
        help="Enable inverse identification of Young's modulus and Poisson's ratio. The default mode keeps them fixed.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume training from an existing checkpoint.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip final field export.")
    return parser.parse_args()


def main():
    args = parse_args()
    case_dir = os.path.abspath(args.case_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The framework core is case agnostic. All single-hole-specific geometry,
    # boundary conditions, and reference conventions are provided by SingleHoleCase.
    case = SingleHoleCase(case_dir, device)
    case.write_metadata(case_dir)

    # The network backbone is another pluggable input of the framework.
    model = build_model(name=args.model_name, activation=args.activation).to(device)

    # The public package defaults to fixed material parameters. Inverse mode can
    # be enabled explicitly from the CLI when parameter identification is needed.
    if args.invert_material:
        material = TrainableMaterialParameters(
            device=device,
            youngs_init=2.0,
            poisson_init=0.2,
        )
    else:
        material = FixedMaterialParameters(
            device=device,
            youngs_modulus=case.reference_youngs_modulus,
            poisson_ratio=case.reference_poissons_ratio,
        )

    trainer = IPINNTrainer(case=case, model=model, material_parameters=material, save_path=case_dir)

    if args.resume:
        trainer.train(resume=True, checkpoint_file=args.resume_checkpoint)
    else:
        trainer.train()

    if not args.skip_eval:
        trainer.load_checkpoint("checkpoint.pth")
        trainer.evaluate()


if __name__ == "__main__":
    main()
