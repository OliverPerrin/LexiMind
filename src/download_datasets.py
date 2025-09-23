import os
import requests
import subprocess

def download_gutenberg():
    """Example: download Pride and Prejudice"""
    url = "https://www.gutenberg.org/files/1342/1342-0.txt"
    os.makedirs("data/raw/books", exist_ok=True)
    out_path = "data/raw/books/pride_and_prejudice.txt"
    if not os.path.exists(out_path):
        r = requests.get(url)
        with open(out_path, "wb") as f:
            f.write(r.content)
    print("Downloaded:", out_path)

# Kaggle dataset download helpers
def download_emotion_dataset():
    """Download the emotions dataset from Kaggle."""
    target_dir = "data/raw/emotion"
    os.makedirs(target_dir, exist_ok=True)
    # Downloading using Kaggle CLI
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "praveengovi/emotions-dataset-for-nlp",
        "-p", target_dir,
        "--unzip"
    ], check=True)
    print("Downloaded Kaggle emotion dataset to", target_dir)

def download_cnn_dailymail():
    """Download the CNN/DailyMail summarization dataset from Kaggle."""
    target_dir = "data/raw/summarization"
    os.makedirs(target_dir, exist_ok=True)
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "gowrishankarp/newspaper-text-summarization-cnn-dailymail",
        "-p", target_dir,
        "--unzip"
    ], check=True)
    print("Downloaded Kaggle CNN/DailyMail dataset to", target_dir)

def download_ag_news():
    """Download the AG News dataset from Kaggle."""
    target_dir = "data/raw/topic"
    os.makedirs(target_dir, exist_ok=True)
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "amananandrai/ag-news-classification-dataset",
        "-p", target_dir,
        "--unzip"
    ], check=True)
    print("Downloaded Kaggle AG News dataset to", target_dir)

if __name__ == "__main__":
    download_gutenberg()
    download_emotion_dataset()
    download_cnn_dailymail()
    download_ag_news()