"""Upload model checkpoint to Hugging Face Hub."""
import os
from huggingface_hub import HfApi, create_repo

# Login uses HF_TOKEN environment variable automatically

# Initialize API
api = HfApi()

# Model repository
repo_id = "OliverPerrin/LexiMind-Model"
model_file = "checkpoints/best.pt"

# Create repository if it doesn't exist
try:
    print(f"Creating model repository {repo_id}...")
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    print("✓ Repository created/verified")
except Exception as e:
    print(f"Repository creation: {e}")

print(f"Uploading {model_file} to {repo_id}...")

# Upload the model file
api.upload_file(
    path_or_fileobj=model_file,
    path_in_repo="best.pt",
    repo_id=repo_id,
    repo_type="model",
)

print("✓ Model uploaded successfully!")
