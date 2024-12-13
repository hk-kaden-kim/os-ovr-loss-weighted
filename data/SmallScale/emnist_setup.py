import os
from pathlib import Path
import argparse

import torchvision
import torchvision.datasets.utils as utils
import torchvision.transforms as transforms

def command_line_options():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--output", "-O", default='/local/scratch/hkim', 
                        help="Root folder for the EMNIST dataset.")

    return parser.parse_args()

def find_gz_files(folder_path):
    gz_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".gz"):
                gz_files.append(os.path.join(root, file))
    return gz_files

if __name__ == "__main__":

    args = command_line_options()
    url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
    output_directory = Path(args.output)

    # Download dataset
    utils.download_and_extract_archive(url, output_directory,
                                       os.path.join(output_directory, 'EMNIST'), 
                                       remove_finished=True)

    # Unzip .gz file
    gz_files = find_gz_files(os.path.join(output_directory,'EMNIST'))
    if len(gz_files) > 0:
        for gz in gz_files:
            res = utils.extract_archive(gz, remove_finished=True)
        print(f"\n...\nExtract {gz} -> {res}")

    # Folder name align
    os.rename(os.path.join(output_directory,'EMNIST','gzip'),
              os.path.join(output_directory,'EMNIST','raw'))

    # Test for loading EMNIST via pytorch
    try:
        mnist = torchvision.datasets.EMNIST(
                    root=output_directory,
                    train=True,
                    download=False, # RuntimeError: File not found or corrupted
                    split="mnist",
                    transform=transforms.ToTensor()
                )
        print(f"\nDONE!\nSet data.smallscale.root to {output_directory} in train.yaml")
    except:
        print("Issue to call EMNIST dataset from torchvision!")

