import argparse
import zipfile
import os
import laspy
import time
import shutil


def main(args):
    if os.path.exists(f"/tmp/lidar/{args.input.split('/')[-1].split('.zip')[0]}"):
        print(f"/tmp/lidar/{args.input.split('/')[-1].split('.zip')[0]} already exists")
        read_las = False
        while read_las == False:
            try:
                laspy.read(f"/tmp/lidar/{args.input.split('/')[-1].split('.zip')[0]}")
                read_las = True
            except ValueError:
                print("Un-zipping not finished. Wait 5 seconds.")
                time.sleep(5)
        time.sleep(5)
        os.remove(f"/tmp/lidar/{args.input.split('/')[-1].split('.zip')[0]}")
        print(f"Removing /tmp/lidar/{args.input.split('/')[-1].split('.zip')[0]}")

        exit(1)

    print(f"Un-zipping to /tmp/lidar/{args.input.split('/')[-1].split('.zip')[0]}")
    total, used, _ = shutil.disk_usage("/")
    while used / total > 0.2:
        print(f"{used*100 / total: .2f}% full")
        time.sleep(60)
        total, used, _ = shutil.disk_usage("/")
    with zipfile.ZipFile(args.input, "r") as zip_ref:
        zip_ref.extractall("/tmp/lidar")

    print("Un-zipping done")
    print("Total: %d GiB" % (total // (2**30)))
    print("Used: %d GiB" % (used // (2**30)))
    print(f"{used*100 / total: .2f}% full")


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to zip file")
    args = parser.parse_args()

    main(args)
