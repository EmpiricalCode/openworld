import modal

app = modal.App.lookup("my-sandbox", create_if_missing=True)

sb = modal.Sandbox.create(
    gpu="a10g",
    timeout=3600 * 24,
    app=app,
)

print(f"Sandbox ID: {sb.object_id}")