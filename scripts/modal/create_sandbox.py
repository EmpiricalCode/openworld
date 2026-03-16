import modal

github_secret = modal.Secret.from_name("github-token")

image = (
    modal.Image.debian_slim()
    .apt_install("git", "tmux", "nano")
    .pip_install("torch", "numpy", "h5py", "gymnasium", "vizdoom", "opencv-python-headless")
)

app = modal.App.lookup("my-sandbox", create_if_missing=True)
data_vol = modal.Volume.from_name("data-deathmatch-large", create_if_missing=True)
tokenizer_vol = modal.Volume.from_name("tokenizer-checkpoint-deathmatch")
lam_vol = modal.Volume.from_name("lam-deathmatch-large")
dynamics_vol = modal.Volume.from_name("dynamics-deathmatch-large")

sb = modal.Sandbox.create(
    image=image,
    gpu="a10",
    timeout=3600 * 24,
    app=app,
    secrets=[github_secret],
    volumes={
        "/data": data_vol,
        "/tokenizer-ckpt": tokenizer_vol,
        "/lam-ckpt": lam_vol,
        "/dynamics-ckpt": dynamics_vol,
    },
)

def run(cmd):
    p = sb.exec("bash", "-c", cmd)
    p.wait()
    stdout = p.stdout.read()
    stderr = p.stderr.read()
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)

run("git clone https://$GITHUB_TOKEN@github.com/EmpiricalCode/dream-rl.git /root/dream-rl")
run("cd /root/dream-rl && python -m venv venv")
run("cd /root/dream-rl && venv/bin/pip install -r requirements.txt")
run("mkdir -p /root/dream-rl/checkpoints && cp /tokenizer-ckpt/* /root/dream-rl/checkpoints/ && cp /lam-ckpt/* /root/dream-rl/checkpoints/ && cp /dynamics-ckpt/* /root/dream-rl/checkpoints/")
run("ls -la /root/dream-rl/checkpoints/")

print(f"Sandbox ID: {sb.object_id}")