import os
import argparse
import subprocess

def main(output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the URL and output file path
    url = "https://www.kaggle.com/api/v1/datasets/download/hearfool/vggface2"
    output_file = os.path.join(output_dir, "vggface2.zip")
    
    # Use curl to download the dataset
    subprocess.run(["curl", "-L", "-o", output_file, url], check=True)
    
    # Ensure the path is absolute
    absolute_path = os.path.abspath(output_file)
    
    print("Path to dataset files:", absolute_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download VGGFace2 dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to download the dataset to")
    args = parser.parse_args()

    main(args.output_dir)
