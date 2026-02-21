"""
Upload inference videos to a Modal volume. Run this inside the sandbox.
Usage: python scripts/modal/upload_to_volume.py --volume dream-rl-outputs
"""
import modal
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume', required=True, help='Modal volume name')
    args = parser.parse_args()

    dir = 'outputs/inference'
    volume = args.volume

    vol = modal.Volume.from_name(volume, create_if_missing=True)

    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    print(f"Found {len(files)} files in {dir}")

    with vol.batch_upload() as batch:
        for filename in files:
            local_path = os.path.join(dir, filename)
            remote_path = f"/inference/{filename}"
            batch.put_file(local_path, remote_path)
            print(f"  Uploaded: {filename} -> {remote_path}")

    print(f"\nDone. Download with: modal volume get {volume} inference/ ./")


if __name__ == '__main__':
    main()
