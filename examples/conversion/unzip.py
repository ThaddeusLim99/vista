import argparse
import zipfile
import os


def main(args):
    if (
        os.path.exists(f"/tmp/lidar/{args.input.split('/')[-1].split('.zip')[0]}")
        == False
    ):
        print(f"/tmp/lidar/{args.input.split('/')[-1].split('.zip')[0]} already exists")
        exit(1)
    print(f"Un-zipping to /tmp/lidar/{args.input.split('/')[-1].split('.zip')[0]}")
    with zipfile.ZipFile(args.input, "r") as zip_ref:
        zip_ref.extractall("/tmp/lidar")
    print("Un-zipping done")


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to zip file")
    args = parser.parse_args()

    main(args)
