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

    root_dir = 'outputs'
    volume = args.volume

    vol = modal.Volume.from_name(volume, create_if_missing=True)

    count = 0
    with vol.batch_upload() as batch:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                local_path = os.path.join(dirpath, filename)
                remote_path = os.path.relpath(local_path, root_dir)
                batch.put_file(local_path, f"/{remote_path}")
                print(f"  Uploaded: {local_path} -> /{remote_path}")
                count += 1

    print(f"\nDone. Uploaded {count} files. Download with: modal volume get {volume} / ./")


if __name__ == '__main__':
    main()
