"""
Download helpers for datasets.

This version:
- Adds robust error handling when Kaggle API is not configured.
- Stores files under data/raw/ subfolders.
- Keeps the Gutenberg direct download example.

Make sure you have Kaggle credentials configured if you call Kaggle downloads.
"""
import os
import requests

def download_gutenberg(out_dir="data/raw/books", gutenberg_id: int = 1342, filename: str = "pride_and_prejudice.txt"):
    """Download a Gutenberg text file by direct URL template (best-effort)."""
    url = f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    if os.path.exists(out_path):
        print("Already downloaded:", out_path)
        return out_path
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        print("Downloaded:", out_path)
        return out_path
    except Exception as e:
        print("Failed to download Gutenberg file:", e)
        return None

# Kaggle helpers: optional, wrapped to avoid hard failure when Kaggle isn't configured.
def _safe_kaggle_download(dataset: str, path: str):
    try:
        import kaggle
    except Exception as e:
        print("Kaggle package not available or not configured. Please install 'kaggle' and configure API token. Error:", e)
        return False
    try:
        os.makedirs(path, exist_ok=True)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset, path=path, unzip=True)
        print(f"Downloaded Kaggle dataset {dataset} to {path}")
        return True
    except Exception as e:
        print("Failed to download Kaggle dataset:", e)
        return False

def download_emotion_dataset():
    target_dir = "data/raw/emotion"
    return _safe_kaggle_download('praveengovi/emotions-dataset-for-nlp', target_dir)

def download_cnn_dailymail():
    target_dir = "data/raw/summarization"
    return _safe_kaggle_download('gowrishankarp/newspaper-text-summarization-cnn-dailymail', target_dir)

def download_ag_news():
    target_dir = "data/raw/topic"
    return _safe_kaggle_download('amananandrai/ag-news-classification-dataset', target_dir)

if __name__ == "__main__":
    download_gutenberg()
    download_emotion_dataset()
    download_cnn_dailymail()
    download_ag_news()