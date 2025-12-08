"""Delete Hugging Face repository."""
from huggingface_hub import HfApi

# Login uses HF_TOKEN environment variable automatically

# Initialize API
api = HfApi()

# Delete the OliverPerrin/LexiMind model repository
repo_id = "OliverPerrin/LexiMind"

try:
    print(f"Deleting model repository {repo_id}...")
    api.delete_repo(repo_id=repo_id, repo_type="model")
    print(f"✓ Successfully deleted {repo_id}")
except Exception as e:
    print(f"Error deleting repository: {e}")
