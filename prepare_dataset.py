#!/usr/bin/env python3
"""
Dataset Preparation Script for Custom ICPR Dataset

This script processes the custom dataset structure and:
1. Reads JSON annotations from each track folder
2. Creates train/validation/test split
3. Generates dataset.txt file in the required format
4. Creates .txt metadata files for each HR image
"""

import os
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


def load_annotations(json_path):
    """Load annotations from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_corner_points(corners_dict):
    """Extract corner points from corners dictionary."""
    # Order: top-left, top-right, bottom-right, bottom-left
    if 'top-left' in corners_dict and 'top-right' in corners_dict and \
       'bottom-right' in corners_dict and 'bottom-left' in corners_dict:
        tl = corners_dict['top-left']
        tr = corners_dict['top-right']
        br = corners_dict['bottom-right']
        bl = corners_dict['bottom-left']
        return f"{tl[0]},{tl[1]} {tr[0]},{tr[1]} {br[0]},{br[1]} {bl[0]},{bl[1]}"
    return "0,0 0,0 0,0 0,0"  # Default if missing


def create_metadata_file(hr_path, annotations, image_name):
    """Create .txt metadata file for HR image in the format expected by the code."""
    txt_path = hr_path.with_suffix('.txt')
    
    plate_text = annotations.get('plate_text', 'UNKNOWN')
    plate_layout = annotations.get('plate_layout', 'Brazilian')
    
    # Get corners for this specific image
    corners_dict = annotations.get('corners', {})
    image_corners = corners_dict.get(image_name, {})
    points = get_corner_points(image_corners)
    
    # Determine type (default to 'car' for Brazilian plates)
    vehicle_type = 'car'  # You can modify this if you have type info in JSON
    
    # Write metadata file in the expected format
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Type: {vehicle_type}\n")
        f.write(f"Plate: {plate_text}\n")
        f.write(f"Layout: {plate_layout}\n")
        f.write(f"Points: {points}\n")
    
    return plate_text, plate_layout, vehicle_type


def process_dataset(dataset_root, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Process the dataset and create required files.
    
    Args:
        dataset_root: Root directory containing track_* folders
        output_dir: Directory to save dataset.txt and metadata files
        train_ratio: Ratio for training set (default: 0.7)
        val_ratio: Ratio for validation set (default: 0.15)
        test_ratio: Ratio for test set (default: 0.15)
        seed: Random seed for reproducibility
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Set random seed
    random.seed(seed)
    
    # Collect all tracks
    track_folders = sorted([d for d in dataset_root.iterdir() 
                           if d.is_dir() and d.name.startswith('track_')])
    
    print(f"Found {len(track_folders)} track folders")
    
    # Collect all image pairs
    image_pairs = []
    
    for track_folder in track_folders:
        json_path = track_folder / 'annotations.json'
        
        if not json_path.exists():
            print(f"Warning: No annotations.json found in {track_folder}")
            continue
        
        annotations = load_annotations(json_path)
        
        # Find all HR images in this track
        hr_images = sorted([f for f in track_folder.iterdir() 
                           if f.is_file() and f.name.startswith('hr-') and f.suffix in ['.png', '.jpg']])
        
        for hr_image in hr_images:
            # Find corresponding LR image
            # hr-001.png -> lr-001.png
            lr_name = hr_image.name.replace('hr-', 'lr-')
            lr_image = track_folder / lr_name
            
            if not lr_image.exists():
                print(f"Warning: LR image not found for {hr_image}")
                continue
            
            image_pairs.append({
                'hr_path': hr_image,
                'lr_path': lr_image,
                'track_folder': track_folder,
                'annotations': annotations,
                'image_name': hr_image.name
            })
    
    print(f"Found {len(image_pairs)} HR-LR image pairs")
    
    # Shuffle and split
    random.shuffle(image_pairs)
    
    n_total = len(image_pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val  # Remaining goes to test
    
    train_pairs = image_pairs[:n_train]
    val_pairs = image_pairs[n_train:n_train + n_val]
    test_pairs = image_pairs[n_train + n_val:]
    
    print(f"\nSplit statistics:")
    print(f"  Training:   {len(train_pairs)} ({len(train_pairs)/n_total*100:.1f}%)")
    print(f"  Validation: {len(val_pairs)} ({len(val_pairs)/n_total*100:.1f}%)")
    print(f"  Testing:    {len(test_pairs)} ({len(test_pairs)/n_total*100:.1f}%)")
    
    # Create metadata files and dataset.txt entries
    dataset_lines = []
    
    for split_name, pairs in [('training', train_pairs), 
                              ('validation', val_pairs), 
                              ('testing', test_pairs)]:
        for pair in pairs:
            hr_path = pair['hr_path']
            lr_path = pair['lr_path']
            annotations = pair['annotations']
            image_name = pair['image_name']
            
            # Create metadata file for HR image
            plate_text, plate_layout, vehicle_type = create_metadata_file(
                hr_path, annotations, image_name
            )
            
            # Add to dataset.txt
            dataset_lines.append(f"{hr_path};{lr_path};{split_name}")
    
    # Write dataset.txt
    dataset_txt_path = output_dir / 'dataset.txt'
    with open(dataset_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(dataset_lines))
    
    print(f"\n✓ Created dataset.txt: {dataset_txt_path}")
    print(f"  Total entries: {len(dataset_lines)}")
    print(f"  Training:   {len(train_pairs)}")
    print(f"  Validation: {len(val_pairs)}")
    print(f"  Testing:    {len(test_pairs)}")
    
    # Also create separate split files for convenience
    for split_name, pairs in [('training', train_pairs), 
                              ('validation', val_pairs), 
                              ('testing', test_pairs)]:
        split_file = output_dir / f'{split_name}.txt'
        split_lines = []
        for pair in pairs:
            hr_path = pair['hr_path']
            lr_path = pair['lr_path']
            split_lines.append(f"{hr_path};{lr_path};{split_name}")
        
        with open(split_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(split_lines))
        print(f"✓ Created {split_name}.txt: {split_file}")
    
    return dataset_txt_path


def main():
    parser = argparse.ArgumentParser(
        description='Prepare custom ICPR dataset for training'
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        required=True,
        help='Root directory containing track_* folders (e.g., /path/to/Brazilian/)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./dataset',
        help='Output directory for dataset.txt and metadata files (default: ./dataset)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Calculate test ratio if not provided
    if args.test_ratio is None:
        args.test_ratio = 1.0 - args.train_ratio - args.val_ratio
    
    print("=" * 60)
    print("Dataset Preparation Script")
    print("=" * 60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    
    dataset_txt_path = process_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    print("\n" + "=" * 60)
    print("Dataset preparation completed successfully!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Use the dataset.txt file for training:")
    print(f"   cd Proposed")
    print(f"   python training.py -t {dataset_txt_path} -s ./checkpoints -b 16 -m 0")
    print(f"\n2. The metadata .txt files have been created alongside your HR images")
    print("=" * 60)


if __name__ == '__main__':
    main()



