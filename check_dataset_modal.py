import modal
import os
from pathlib import Path

app = modal.App("lpr-rsr-ext")

# Define a Volume to store logs and checkpoints
results_vol = modal.Volume.from_name("lpr-rsr-ext", create_if_missing=True)

@app.function(
    image=modal.Image.debian_slim().pip_install(["pydantic"]),  # Minimal image
    volumes={"/root/experiments": results_vol},
    timeout=300
)
def check_dataset():
    """Check dataset structure in Modal volume."""
    dataset_root = Path("/root/experiments/ICPR_dataset/Brazilian")

    print(f"Dataset root exists: {dataset_root.exists()}")

    if dataset_root.exists():
        # Count track folders
        track_folders = sorted([d for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith('track_')])
        print(f"Found {len(track_folders)} track folders")

        # Check first few tracks
        for track in track_folders[:3]:
            print(f"\nTrack: {track.name}")
            hr_files = sorted([f for f in track.iterdir() if f.name.startswith('hr-') and f.suffix in ['.png', '.jpg']])
            print(f"  HR files: {len(hr_files)}")

            if hr_files:
                # Check first HR file
                first_hr = hr_files[0]
                txt_file = track / (first_hr.stem + '.txt')
                print(f"  First HR: {first_hr.name}")
                print(f"  Txt file exists: {txt_file.exists()}")

                if txt_file.exists():
                    print(f"  Txt content:")
                    with open(txt_file, 'r') as f:
                        for i, line in enumerate(f.readlines(), 1):
                            print(f"    {i}: {line.rstrip()}")

    # Check if dataset.txt exists
    dataset_file = Path("/__modal/volumes/vo-u3Z2Cgd39jj27tH81zj6ZJ/lpr-rsr-ext/dataset/dataset_modal.txt")
    print(f"\nDataset modal.txt exists: {dataset_file.exists()}")

    if dataset_file.exists():
        with open(dataset_file, 'r') as f:
            lines = f.readlines()
            print(f"Dataset.txt lines: {len(lines)}")
            print(f"First line: {lines[0] if lines else 'None'}")

@app.local_entrypoint()
def main():
    print("Checking dataset in Modal volume...")
    check_dataset.remote()
