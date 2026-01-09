import os
import argparse
from pathlib import Path
from __syslog__ import EventlogHandler


def _add_wandb_args(parser, include_training_args=False):
    """Add common W&B arguments to parser."""
    parser.add_argument("--wandb-project", type=str, default="lpr-super-resolution",
                        help="Weights & Biases project name")
    parser.add_argument("--disable-wandb", action="store_true", default=False,
                        help="Disable Weights & Biases logging")

    if include_training_args:
        parser.add_argument("--wandb-run-name", type=str, default=None,
                            help="Custom run name for W&B (auto-generated if not provided)")
        parser.add_argument("--wandb-entity", type=str, default=None,
                            help="W&B entity/team name (optional)")
        parser.add_argument("--wandb-log-interval", type=int, default=5,
                            help="Log sample images every N epochs (default: 5)")
    else:
        parser.add_argument("--wandb-run-id", type=str, default=None,
                            help="W&B run ID to resume (from training)")


@EventlogHandler
def args_m01():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--samples", type=Path, required=True, help = "Path containing the samples")
    ap.add_argument("-s", "--save", type=Path, required=True, help="Backup and saving path")
    ap.add_argument("-b", "--batch", type=int, required=True, help="Batch size")
    ap.add_argument("-m", "--mode", type=int, required=True, choices = [0, 1],  help='Reset Model, Load Model, Evaluate Model. Reset=0, Load=1, Evaluate=2')
    ap.add_argument("--model", type=Path, required=False, help="Path containing the model", default=Path(''))
    ap.add_argument("--debug", action="store_true", default=False,
                    help="Enable debug mode: use tiny dataset subset and run 2-3 epochs")
    ap.add_argument("--debug-samples", type=int, default=20,
                    help="Number of training samples to use in debug mode (default: 20)")

    _add_wandb_args(ap, include_training_args=True)

    try:
        args = ap.parse_args()
    except:
        ap.print_help()
        raise Exception("Missing Arguments")
        
    if not(args.samples.is_file() and args.samples.suffix == '.txt'):
        raise argparse.ArgumentTypeError("\'-samples;--path_samples\' file not found.")
    
    if args.mode == 0:
        if not args.save.is_dir():
            print(f'Creating save directory: {args.save}')
            args.save.mkdir(parents=True, exist_ok=True)
        args.model = None
    
    elif args.mode == 1:
        if not(args.model.is_file() and args.model.suffix == '.pt'):
            raise argparse.ArgumentTypeError("for \'--m;--mode\' set to 1 \'--model;--path_model\' must be a valid file.")
            
            
        if not args.save.exists():
            raise argparse.ArgumentTypeError("the \'--save;--save_model\' is necessary.")
            
    return args
        
def args_m2():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--samples", type=Path, required=True, help = "Path containing the samples")
    ap.add_argument("-s", "--save", type=Path, required=True, help="Backup and saving path")
    ap.add_argument("-b", "--batch", type=int, required=True, help="Batch size")
    ap.add_argument("--model", type=Path, required=True, help="Path containing the model", default=Path(''))

    _add_wandb_args(ap, include_training_args=False)

    try:
        args = ap.parse_args()
    except:
        ap.print_help()
        raise Exception("Missing Arguments")
        
    if not args.save.is_dir():
        print(f'Creating evaluation directory: {args.save}')
        args.save.mkdir(parents=True, exist_ok=True)
    
    if not(args.model.is_file() and args.model.suffix == '.pt'):
        raise argparse.ArgumentTypeError("for \'--m;--mode\' set to 1 \'--model;--path_model\' must be a valid file.")
    
    return args    
    
if __name__ == "__main__":
    args_m2()
    # args_m01()