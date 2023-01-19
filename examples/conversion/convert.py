import h5py
import argparse
import laspy
import numpy as np
import zipfile
import math
import os
from statistics import mean
import pandas as pd


def main(args):
    if args.input.endswith(".zip"):
        with zipfile.ZipFile(args.input, "r") as zip_ref:
            zip_ref.extractall("/tmp/lidar")

        args.input = f"/tmp/lidar/{args.input.split('/')[-1].split('.zip')[0]}"

        print(f"/tmp/lidar/{args.input.split('/')[-1].split('.zip')[0]}")

    las = laspy.read(args.input)

    road_points = pd.read_csv(
        "./examples/Trajectory/road_points.csv", sep=",", header=None
    ).values
    forwards = pd.read_csv(
        "./examples/Trajectory/forwards.csv", sep=",", header=None
    ).values
    leftwards = pd.read_csv(
        "./examples/Trajectory/leftwards.csv", sep=",", header=None
    ).values

    trajectory = road_points * 1000

    i = args.frame
    print(f"Frame #: {i}")

    pov_X = trajectory[i][0]
    pov_Y = trajectory[i][1] - 5000000000
    pov_Z = trajectory[i][2]

    x = np.array(las.X)
    y = np.array(las.Y)
    z = np.array(las.Z)

    xyz = np.asarray([x, y, z], dtype=np.float64).T

    # Traslantion
    xyz -= np.array([pov_X, pov_Y, pov_Z])

    xyz_distance = np.sqrt(
        np.square(xyz[:, 0]) + np.square(xyz[:, 1]) + np.square(xyz[:, 2])
    )

    indices = np.where((xyz_distance < 245000))
    xyz = xyz[indices]

    # Rotation 1
    cos_1 = forwards[i][0] / ((forwards[i][0] ** 2 + forwards[i][1] ** 2) ** (0.5))
    sin_1 = forwards[i][1] / ((forwards[i][0] ** 2 + forwards[i][1] ** 2) ** (0.5))
    xyz = np.dot(np.array([[cos_1, sin_1, 0], [-sin_1, cos_1, 0], [0, 0, 1]]), xyz.T).T

    # Rotation 2
    cos_2 = ((forwards[i][0] ** 2 + forwards[i][1] ** 2) ** (0.5)) / (
        ((forwards[i][0] ** 2 + forwards[i][1] ** 2) + forwards[i][2] ** 2) ** (0.5)
    )
    sin_2 = forwards[i][2] / (
        ((forwards[i][0] ** 2 + forwards[i][1] ** 2) + forwards[i][2] ** 2) ** (0.5)
    )
    xyz = np.dot(np.array([[cos_2, 0, -sin_2], [0, 1, 0], [sin_2, 0, cos_2]]), xyz.T).T

    tangent = leftwards[i][2] / ((leftwards[i][0] ** 2 + leftwards[i][1] ** 2) ** (0.5))
    cross_angle = math.atan(tangent)
    cos_3 = math.cos(cross_angle)
    sin_3 = math.sin(cross_angle)

    xyz = np.dot(np.array([[1, 0, 0], [0, cos_3, -sin_3], [0, sin_3, cos_3]]), xyz.T).T

    # Sensor at 1.2 meter above
    xyz[:, 2] -= 1200

    with h5py.File("./examples/vista_traces/lidar/lidar_3d.h5", "w") as f:
        f["timestamp"] = [[0], [0.1], [0.2]]
        f["xyz"] = [xyz]
        f["intensity"] = [las.intensity[indices]]

    f2 = h5py.File("./examples/vista_traces/lidar/lidar_3d.h5", "r")
    print(f2["timestamp"])
    print(f2["xyz"])
    print(f2["intensity"])

    samples = np.random.rand(1024, 3)
    samples[:, 0] = samples[:, 0] * 150000 + 95000
    samples[:, 1] = (samples[:, 1] - 0.5) * 100000
    samples[:, 2] = (samples[:, 2] - 0.5) * 30000

    pd.DataFrame(xyz).to_csv("./examples/vista_traces/lidar/lidar_3d.csv")

    try:
        with open("/tmp/lidar/trajectory.csv", "a") as f:
            f.write(f"{pov_X}, {pov_Y}, {pov_Z}, {sin_1}, {cos_1}, {sin_2}, {cos_2}\n")
    except FileNotFoundError:
        os.mkdir("/tmp/lidar")
        with open("/tmp/lidar/trajectory.csv", "w") as f:
            f.write(f"{pov_X}, {pov_Y}, {pov_Z}, {sin_1}, {cos_1}, {sin_2}, {cos_2}\n")


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to .las file to convert to .h5")
    parser.add_argument("--frame", type=int, help="Frame number")
    args = parser.parse_args()

    main(args)
