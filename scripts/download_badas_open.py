from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nexar-ai/BADAS-Open",
    local_dir="./models/BADAS-Open",
    token=True
)

print("BADAS-Open downloaded to ./models/BADAS-Open")
