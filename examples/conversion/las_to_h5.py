import h5py
import argparse
import laspy
import numpy as np


def main(args):
    las = laspy.read(args.input)
    x = np.array([las.X])
    y = np.array([las.Y])
    z = np.array([las.Z])
    x = x - np.average(x)
    y = y - np.average(y)
    z = z - np.average(z)
    xyz = np.array([x, y, z]).T
    np.savetxt(
        "./examples/vista_traces/lidar/input.csv",
        xyz.reshape([len(xyz), 3]),
        delimiter=",",
        fmt="%f",
    )

    with h5py.File(".".join(args.input.split(".")[:-1]) + ".h5", "w") as f:
        f["timestamp"] = [[0], [0.1], [0.2]]
        f["xyz"] = [xyz]
        f["intensity"] = [las.intensity]

    f2 = h5py.File(".".join(args.input.split(".")[:-1]) + ".h5", "r")
    print(f2["timestamp"])
    print(f2["xyz"])
    print(f2["intensity"])


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to .las file to convert to .h5")
    args = parser.parse_args()

    main(args)
