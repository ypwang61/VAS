import argparse
import os
from pathlib import Path

import torch

from baselines.apply_filter import apply_filter

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

BASELINES = {
    "no_filter",
    "basic_filter",
    "text_based",
    "image_based",
    "image_based_intersect_clip_score",
    "clip_score",
    "laion2b",
    
    "vas",
    "vas_d",
}

ARCH = {
    "b32",
    "l14",
}

CLUSTER_CENTROID_SCALES = [
    "small",
    "medium",
    "large",
    "xlarge",
]


def check_args(args):
    if args.name not in BASELINES:
        raise ValueError(f"--name must be in: {BASELINES}")

    if args.name == "laion2b":
        if (
            args.fraction is not None
            or args.threshold is not None
            or args.arch is not None
            or args.image_based_scale is not None
        ):
            raise ValueError("laion2b does not support clip_score or image_based flags")

    # clip_score checks
    if "clip_score" in args.name:
        if args.fraction is None and args.threshold is None:
            raise ValueError(
                "--fraction or --threshold must be specified for clip_score baselines"
            )
        if args.fraction is not None and args.threshold is not None:
            raise ValueError(
                "specify either --fraction or --threshold for clip_score baselines but not both"
            )
        if args.arch is None:
            raise ValueError(f"specify architecture {ARCH}, for clip_score baselines")
    if args.fraction is not None and not ("clip_score" in args.name or "vas" in args.name):
        raise ValueError("--fraction value only used for clip_score / vas baselines")
    if args.threshold is not None and not ("clip_score" in args.name or "vas" in args.name):
        raise ValueError("--threshold value only used for clip_score / vas baselines")
    if args.arch is not None and not ("clip_score" in args.name or "vas" in args.name or "vis_data" in args.name):
        raise ValueError("--arch value only used for clip_score / vas baselines")
    
    # vas checks
    if "vas" in args.name:
        if args.fraction_vas is None and args.threshold_vas is None:
            raise ValueError(
                "--fraction_vas or --threshold_vas must be specified for vas baselines"
            )
        if args.fraction_vas is not None and args.threshold_vas is not None:
            raise ValueError(
                "specify either --fraction_vas or --threshold_vas for vas baselines but not both"
            )
        if args.arch is None:
            raise ValueError(f"specify architecture {ARCH}, for clip score filter first")
    
        if "vas_d" in args.name:
            assert args.fraction_vas is not None, "fraction_vas must be specified for vas_d baselines"
            assert args.num_iters is not None, "num_iters must be specified for vas_d baselines"
        
    
    # image_based checks
    if args.image_based_scale is None and "image_based" in args.name:
        raise ValueError(
            "--image_based_scale value must be passed for image_based and image_based_intersect_clip_score_* baselines (for clustering)"
        )
    if args.image_based_scale is not None and not ("image_based" in args.name):
        raise ValueError(
            "--image_based_scale should only be passed for image_based and image_based_intersect_clip_score_* baselines (for clustering)"
        )
    if "image_based" in args.name and not torch.cuda.is_available():
        raise ValueError(
            "gpus needed for image_based baselines, torch.cuda.is_available() must return true"
        )
    if args.given_uids_path is not None:
        if args.fraction is not None:
            raise ValueError(
                "fraction value should not be passed when given_uids_path is passed"
            )

    npy_parent = Path(args.save_path).parent
    if not os.path.exists(npy_parent):
        print(f"creating: {npy_parent}")
        os.mkdir(npy_parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This is a command line script for reproducing the main DataComp filtering baselines. The output of the script is a numpy file (.npy) containing the uids in the filtered subsets in sorted binary format. Please see README.md for additional information"
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        choices=list(BASELINES),
        help="name of the baseline",
    )

    parser.add_argument(
        "--metadata_dir",
        type=str,
        required=True,
        help="directory (local or cloud) containing parquet, npz metadata",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="path to output .npy, note: cloudpaths are not supported for this arg",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        default=None,
        help="A threshold to apply on a CLIP score (e.g., a value of 0.25 will only keep examples with CLIP score over 0.25)",
    )

    parser.add_argument(
        "--fraction",
        type=float,
        required=False,
        default=None,
        help="a fraction of metadata to keep according to CLIP score (e.g., a value of 0.25 will keep the top 25 percent of examples in the pool by CLIP score)",
    )

    parser.add_argument(
        "--arch",
        type=str,
        required=False,
        choices=list(ARCH),
        help="kinds of features (b32 or l14) on which to run the CLIP score filter",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=os.cpu_count(),
        help="number of workers, generally set to number of cpu cores. workers read their metadata files and filter them in parallel).",
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        required=False,
        default=torch.cuda.device_count(),
        help="number of gpus for the image_based gpu implementation. num_gpus metadata files are processed in parallel on each gpu worker. NOTE: this parameter is ignored for non-image_basesd baselines",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=1024,
        help="batch size for the image_based gpu implementation. NOTE: this parameter is ignored for non-image_basesd baselines",
    )

    parser.add_argument(
        "--image_based_scale",
        type=str,
        required=False,
        choices=CLUSTER_CENTROID_SCALES,
        help="datacomp scale, used for the clutering baselines",
        default=None,
    )

    parser.add_argument(
        "--centroid_path",
        type=str,
        required=False,
        help="path to the centroid npy file, used for the clutering baselines",
        default=None,
    )
    
    
    parser.add_argument(
        "--files_path",
        type = str,
        required = True,
        help = "path to the files",
        default=None,
    )
    
    parser.add_argument(
        "--threshold_vas",
        type = float,
        required = False,
        help = "threshold value for VAS",
        default=None,
    )
    
    parser.add_argument(
        "--fraction_vas",
        type = float,
        required = False,
        help = "fraction value for VAS",
        default=None,
    )
    
    parser.add_argument(
        "--target_variance_name",
        type = str,
        required = False,
        help = "target variance name for VAS, like 'imagenet_1k' ",
        default=None,
    )
    
    parser.add_argument(
        "--cache_path",
        type=str,
        required=False,
        help="path to cache the data",
        default=None,
    )
    
    parser.add_argument(
        "--given_uids_path",
        type=str,
        required=False,
        help="path to the given uids, can be used when need to combine our method with other filtering method", 
        default=None,
    )
    
    parser.add_argument(
        "--num_iters",
        type = int,
        required=False,
        help="number of iterations for VAS-D",
    )
    
    parser.add_argument(
        "--batch_size_vass",
        type = int,
        required=False,
        help="batch size for calculate the score of VAS-D. batch_size will be used to calcualte the self variance matrix",
    )
    
    parser.add_argument(
        "--higher_is_better_vas",
        type = int,
        required=False,
        help="if higher is better for VAS",
        default = 1
    )
    
    
    args = parser.parse_args()
    # print args
    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")
    # all error checking happens here and apply_filter assumes correct input
    check_args(args)

    # route the args to the correct baseline
    apply_filter(args)
