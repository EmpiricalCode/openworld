"""
Upload files from a Modal sandbox to a Modal volume.
Usage: modal run scripts/modal/upload_to_volume.py --sandbox-id <id> --remote-dir /root/dream-rl/outputs/inference
"""
import modal
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sandbox-id', required=True, help='Modal sandbox ID')
    parser.add_argument('--remote-dir', default='/root/dream-rl/outputs/inference', help='Directory in sandbox to upload from')
    parser.add_argument('--volume', default='dream-rl-outputs', help='Modal volume name')
    args = parser.parse_args()

    sb = modal.Sandbox.from_id(args.sandbox_id)
    vol = modal.Volume.from_name(args.volume, create_if_missing=True)

    # List files in sandbox dir
    p = sb.exec("bash", "-c", f"ls {args.remote_dir}")
    p.wait()
    files = p.stdout.read().strip().split('\n')
    files = [f for f in files if f]

    print(f"Found {len(files)} files in {args.remote_dir}")

    with vol.batch_upload() as batch:
        for filename in files:
            remote_path = f"{args.remote_dir}/{filename}"
            # Read file from sandbox
            p = sb.exec("bash", "-c", f"base64 {remote_path}")
            p.wait()
            b64 = p.stdout.read().strip()

            import base64, tempfile, os
            data = base64.b64decode(b64)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=filename)
            tmp.write(data)
            tmp.close()

            vol_path = f"/inference/{filename}"
            batch.put_file(tmp.name, vol_path)
            os.unlink(tmp.name)
            print(f"  Uploaded: {filename} -> {vol_path}")

    print(f"\nDone. Download with: modal volume get {args.volume} inference/ ./")


if __name__ == '__main__':
    main()
