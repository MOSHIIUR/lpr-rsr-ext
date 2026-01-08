#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Weights & Biases logging wrapper for LPR Super-Resolution
Provides optional W&B integration with graceful fallback
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
import numpy as np
from PIL import Image

# Try to import wandb, set flag if unavailable
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class WandbLogger:
    """
    Wrapper for Weights & Biases logging with graceful fallback.

    If wandb is not installed or disabled via args, all logging
    operations become no-ops.
    """

    def __init__(self, args, mode='train'):
        """
        Initialize wandb logger.

        Args:
            args: Parsed arguments from __parser__
            mode: 'train' or 'test'
        """
        self.enabled = False
        self.run = None
        self.mode = mode
        self.step_counter = 0

        # Check if wandb should be enabled
        if hasattr(args, 'disable_wandb') and args.disable_wandb:
            print("W&B logging disabled via --disable-wandb flag")
            return

        if not WANDB_AVAILABLE:
            print("⚠️  wandb not installed. Install with: pip install wandb")
            print("   Continuing without W&B logging...")
            return

        self.enabled = True
        self._init_wandb(args)

    def _init_wandb(self, args):
        """Initialize wandb run (training or testing)"""
        if self.mode == 'train':
            self._init_training_run(args)
        elif self.mode == 'test':
            self._init_testing_run(args)

    def _init_training_run(self, args):
        """Initialize wandb for training"""
        # Prepare wandb config
        config = {
            'batch_size': args.batch,
            'learning_rate': 0.0001,  # initial_G_lr from training.py
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau',
            'scheduler_factor': 0.8,
            'scheduler_patience': 2,
            'early_stopping_patience': 20,
            'max_epochs': 3 if (hasattr(args, 'debug') and args.debug) else 200,
            'loss_function': 'MSE + OCR_Features',
            'dataset_path': args.samples.as_posix(),
            'model_architecture': 'RDN + TFAM + AutoEncoder',
            'debug_mode': hasattr(args, 'debug') and args.debug,
        }

        # Add optional debug parameters
        if hasattr(args, 'debug') and args.debug:
            config['debug_samples'] = args.debug_samples if hasattr(args, 'debug_samples') else None

        # Initialize wandb
        self.run = wandb.init(
            project=args.wandb_project if hasattr(args, 'wandb_project') else 'lpr-super-resolution',
            name=args.wandb_run_name if hasattr(args, 'wandb_run_name') else None,
            entity=args.wandb_entity if hasattr(args, 'wandb_entity') else None,
            config=config,
            resume='allow',
        )

        print(f"✅ W&B initialized: {self.run.url}")
        print(f"   Run ID: {self.run.id}")

        # Save run ID to file for testing phase
        run_id_file = Path(args.save) / 'wandb_run_id.txt'
        with open(run_id_file, 'w') as f:
            f.write(self.run.id)
        print(f"   Run ID saved to {run_id_file}")

    def _init_testing_run(self, args):
        """Initialize or resume wandb for testing"""
        run_id = None

        # Try to get run_id from CLI argument
        if hasattr(args, 'wandb_run_id') and args.wandb_run_id:
            run_id = args.wandb_run_id
        else:
            # Try to load from checkpoint directory
            checkpoint_dir = args.model.parent
            run_id_file = checkpoint_dir / 'wandb_run_id.txt'
            if run_id_file.exists():
                with open(run_id_file, 'r') as f:
                    run_id = f.read().strip()
                print(f"📂 Found W&B run ID in {run_id_file}: {run_id}")

        if run_id:
            # Resume existing run
            self.run = wandb.init(
                project=args.wandb_project if hasattr(args, 'wandb_project') else 'lpr-super-resolution',
                id=run_id,
                resume='allow',
            )
            print(f"✅ W&B resumed: {self.run.url}")
        else:
            print("⚠️  No W&B run ID found. Starting new run for testing.")
            config = {
                'mode': 'test_only',
                'batch_size': args.batch,
                'model_path': args.model.as_posix(),
            }
            self.run = wandb.init(
                project=args.wandb_project if hasattr(args, 'wandb_project') else 'lpr-super-resolution',
                config=config,
            )
            print(f"✅ W&B initialized: {self.run.url}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to wandb.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number (uses internal counter if not provided)
        """
        if not self.enabled or self.run is None:
            return

        if step is None:
            step = self.step_counter
            self.step_counter += 1

        wandb.log(metrics, step=step)

    def log_images(self, images: Dict[str, Any], step: Optional[int] = None):
        """
        Log images to wandb.

        Args:
            images: Dict mapping image names to image data
                   Image data can be: PIL Image, numpy array, torch tensor, or wandb.Image
            step: Optional step number
        """
        if not self.enabled or self.run is None:
            return

        wandb_images = {}
        for key, img in images.items():
            if isinstance(img, wandb.Image):
                wandb_images[key] = img
            elif isinstance(img, Image.Image):
                wandb_images[key] = wandb.Image(img)
            elif isinstance(img, np.ndarray):
                wandb_images[key] = wandb.Image(img)
            elif isinstance(img, torch.Tensor):
                # Convert tensor to numpy
                img_np = img.detach().cpu().numpy()
                # If in CHW format, convert to HWC
                if img_np.ndim == 3 and img_np.shape[0] == 3:
                    img_np = np.transpose(img_np, (1, 2, 0))
                wandb_images[key] = wandb.Image(img_np)

        if step is not None:
            wandb.log(wandb_images, step=step)
        else:
            wandb.log(wandb_images)

    def log_table(self, table_name: str, data: List[List], columns: List[str]):
        """
        Log a table to wandb.

        Args:
            table_name: Name of the table
            data: List of rows (each row is a list of values)
            columns: List of column names
        """
        if not self.enabled or self.run is None:
            return

        table = wandb.Table(columns=columns, data=data)
        wandb.log({table_name: table})

    def finish(self):
        """Finish wandb run"""
        if self.enabled and self.run is not None:
            wandb.finish()
            print("✅ W&B run finished")


def get_wandb_logger(args, mode='train') -> WandbLogger:
    """
    Factory function to create WandbLogger.

    Args:
        args: Parsed arguments
        mode: 'train' or 'test'

    Returns:
        WandbLogger instance
    """
    return WandbLogger(args, mode=mode)
