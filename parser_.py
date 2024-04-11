from argparse import ArgumentParser
import os

# parser
def get_args():
    parser = ArgumentParser()
    # standard arguments
    parser.add_argument(
        "--num-workers", type=int, help="# data loading workers", default=16
    )
    parser.add_argument("--batch-size", type=int, default=128, help="input batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--inner-lr", type=float, default=0, help="Learning Rate")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Max Epochs"
    ) 

    parser.add_argument(
        "--root",
        type=str,
        help="",
        default="runs/version_1_feat/",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="utk|fairface|celeba",
        default="utk",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        help="Optimizer Adam|SGD (default: Adam)",
        default="Adam",
    )
    parser.add_argument(
        "--pretrained-teacher",
        type=str,
        metavar="PATH",
        default="",
        help="which checkpoint do you want to start from?",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random Seed for reproducibility",
    )

    parser.add_argument(
        "--mmdv2",
        action="store_true",
    )
    parser.add_argument(
        "--meta2",
        action="store_true",
    )
    parser.add_argument(
        "--meta-mmd",
        action="store_true",
    )
    parser.add_argument(
        "--meta-adv",
        action="store_true",
    )
    parser.add_argument(
        "--eq-odds-ab",
        action="store_true",
    )
    parser.add_argument(
        "--sigmoid",
        action="store_true",
    )
    parser.add_argument("--test", type=str, default="")
    parser.add_argument("--later", type=int, default=0)
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs/",
        help="Log Directory (root, seed, name) only provide name",
    )
    parser.add_argument(
        "--num-meta",
        type=float,
        default=0.05,
        help="% train set for meta step",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="% train set for meta step",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Alpha for scaling KD loss",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1e3,
        help="Beta for scaling KD loss",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=4,
        help="Temp for KD loss",
    )
    parser.add_argument(
        "--lambda_",
        type=float,
        default=3,
        help="Bias Loss weight",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="",
    )
    parser.add_argument(
        "--adv",
        action="store_true",
    )
    parser.add_argument(
        "--student",
        type=str,
        default="",
        help="mobilenetv2 | shufflenetv2 | cnn | alexnet",
    )
    parser.add_argument("--teacher", type=str, default="", help="resnet18 | resnet50")
    parser.add_argument(
        "--base", type=str, default="", help="model for vanilla training"
    )

    parser.add_argument("--fair", action="store_true", help="higher version")
    parser.add_argument(
        "--use-pretrained", action="store_true", help="use pretrained imagenet weights"
    )

    parser.add_argument("--only-gender", action="store_true", help="")
    parser.add_argument("--only-race", action="store_false", help="")
    parser.add_argument("--sub", action="store_true", help="")
    parser.add_argument("--loss-race", action="store_true", help="")
    
    parser.add_argument(
        "--distill",
        action="store_true",
        help="do logit distillation (provide teacher path)",
    )
    parser.add_argument(
        "--AT",
        action="store_true",
        help="do feature distillation (provide teacher path)",
    )
    parser.add_argument(
        "--AT-only",
        action="store_true",
        help="do feature distillation only (provide teacher path)",
    )
    # reference https://github.com/yoshitomo-matsubara/torchdistill/blob/bdb78763cafb1c8bcaea491516fe649fb47398b4/configs/sample/ilsvrc2012/fitnet/resnet18_from_resnet152.yaml#L76
    parser.add_argument(
        "--fitnet-s1",
        action="store_true",
        help="do feature distillation only (provide teacher path)",
    )
    
    parser.add_argument(
        "--fitnet-s2",
        action="store_true",
        help="do feature distillation only (provide teacher path)",
    )
    
    parser.add_argument(
        "--continue-student",
        type=str,
        default=""
    )
    
    parser.add_argument(
        "--get-complexity",
        action="store_true",
    )
    
    parser.add_argument(
        "--fwd-loss",
        action="store_true",
    )
    
    parser.add_argument(
        "--outer-loss",
        type=str,
        default=""
    )
    
    parser.add_argument(
        "--outer-loss-sum",
        type=str,
        default=""
    )
    
    parser.add_argument("--num-task-classes", type=int, default=3)
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help="Value from 1-4 to decide what level of features to distill",
    )

    parser.add_argument(
        "--version2",
        action="store_false",
    )
    parser.add_argument(
        "--vlm",
        type=str,
        help="clip",
        default="",
    )
    
    # dataset
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="",
    )
    parser.add_argument("--datasetv2", action="store_false")

    # other hparams when and if needed
    args = parser.parse_args()

    return args
