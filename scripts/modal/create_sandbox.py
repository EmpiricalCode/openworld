import modal

github_secret = modal.Secret.from_name("github-token")

image = (
    modal.Image.debian_slim()
    .apt_install("git", "tmux", "nano")
    .pip_install("torch", "numpy", "h5py", "gymnasium", "vizdoom", "opencv-python-headless")
)

app = modal.App.lookup("my-sandbox", create_if_missing=True)

sb = modal.Sandbox.create(
    image=image,
    gpu="l40s",
    timeout=3600 * 24,
    app=app,
    secrets=[github_secret],
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
run("cd /root/dream-rl && venv/bin/pip install gdown")
run("mkdir -p /root/dream-rl/data/vizdoom_healthgathering")
run("cd /root/dream-rl && venv/bin/gdown 1NOvGmrsH10UysL-g1zYOktL-hYkRzMWS -O data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5")

print(f"Sandbox ID: {sb.object_id}")