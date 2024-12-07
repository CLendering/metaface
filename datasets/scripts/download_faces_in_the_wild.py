import os
import tarfile
import urllib.request
from datetime import datetime


def download_and_extract(url, download_path, extract_path):
    # Download the file from the URL
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, download_path)
    print(f"Downloaded to {download_path}")

    # Extract the tar.gz file
    if tarfile.is_tarfile(download_path):
        print(f"Extracting {download_path}...")
        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print(f"Extracted to {extract_path}")
    else:
        print(f"{download_path} is not a valid tar.gz file")


if __name__ == "__main__":
    dataset_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    download_path = "/home/camile-lendering/metaface/datasets/lfw.tgz"

    # Create a new folder with a timestamp in the datasets directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extract_path = f"/home/camile-lendering/metaface/datasets/lfw_{timestamp}"

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    os.makedirs(extract_path, exist_ok=True)

    download_and_extract(dataset_url, download_path, extract_path)
