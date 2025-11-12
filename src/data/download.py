"""Dataset download helpers."""

import socket
from pathlib import Path
from subprocess import CalledProcessError, run
from urllib.error import URLError
from urllib.request import urlopen


DOWNLOAD_TIMEOUT = 60


def kaggle_download(dataset: str, output_dir: str) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    try:
        run([
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset,
            "-p",
            str(target),
            "--unzip",
        ], check=True)
    except CalledProcessError as error:
        raise RuntimeError(
            "Kaggle download failed. Verify that the Kaggle CLI is authenticated,"
            " you have accepted the dataset terms on kaggle.com, and your kaggle.json"
            " credentials are located in %USERPROFILE%/.kaggle."
        ) from error


def gutenberg_download(url: str, output_path: str) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response, target.open("wb") as handle:
            chunk = response.read(8192)
            while chunk:
                handle.write(chunk)
                chunk = response.read(8192)
    except (URLError, socket.timeout, OSError) as error:
        raise RuntimeError(f"Failed to download '{url}' to '{target}': {error}") from error
