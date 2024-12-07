import gdown
import os

# Constants
URL = "https://drive.google.com/uc?id=1f_ez-ll6wxDXScrG4ceStZRuLT_K8Uy8"
ZIP_OUTPUT = os.path.join("datasets", "demogpairs.zip")
EXTRACT_OUTPUT = os.path.join("datasets", "demogpairs")


def download_file(url, zip_output, extract_output, unzip=False):
    try:
        # Download the file
        gdown.download(url, zip_output, quiet=False)
        print(f"Downloaded dataset to {zip_output}")

        # Unzip the file if the flag is set
        if unzip:
            import zipfile

            with zipfile.ZipFile(zip_output, "r") as zip_ref:
                zip_ref.extractall(extract_output)
            print(f"Extracted dataset to {extract_output}")
            os.remove(zip_output)
            print(f"Removed zip file {zip_output}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    download_file(URL, ZIP_OUTPUT, EXTRACT_OUTPUT, unzip=True)
