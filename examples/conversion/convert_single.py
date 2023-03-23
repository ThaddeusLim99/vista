import h5py
import argparse
import laspy
import numpy as np
import zipfile
import math
import os
import torch
import pandas as pd


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
    print(device)

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

    trajectory = road_points
    i = args.frame
    print(f"Frame #: {i}")

    pov_X = (trajectory[i][0] - 617000) * 1000
    pov_Y = (trajectory[i][1] - 5658000) * 1000
    pov_Z = (trajectory[i][2]) * 1000

    x = np.array(las.X)
    y = np.array(las.Y)
    z = np.array(las.Z)

    xyz = np.asarray([x, y, z], dtype=np.float64).T / 100

    # Traslantion
    xyz -= np.array([pov_X, pov_Y, pov_Z])

    xyz_distance = np.sqrt(
        np.square(xyz[:, 0]) + np.square(xyz[:, 1]) + np.square(xyz[:, 2])
    )

    indices = np.where((xyz_distance < args.range * 1000))
    xyz = torch.tensor(xyz[indices]).to(device)

    # Rotation 1
    cos_1 = forwards[i][0] / ((forwards[i][0] ** 2 + forwards[i][1] ** 2) ** (0.5))
    sin_1 = forwards[i][1] / ((forwards[i][0] ** 2 + forwards[i][1] ** 2) ** (0.5))
    xyz = torch.matmul(
        torch.tensor([[cos_1, sin_1, 0], [-sin_1, cos_1, 0], [0, 0, 1]])
        .double()
        .to(device),
        xyz.double().T,
    ).T

    # Rotation 2
    cos_2 = ((forwards[i][0] ** 2 + forwards[i][1] ** 2) ** (0.5)) / (
        ((forwards[i][0] ** 2 + forwards[i][1] ** 2) + forwards[i][2] ** 2) ** (0.5)
    )
    sin_2 = forwards[i][2] / (
        ((forwards[i][0] ** 2 + forwards[i][1] ** 2) + forwards[i][2] ** 2) ** (0.5)
    )
    xyz = torch.matmul(
        torch.tensor([[cos_2, 0, -sin_2], [0, 1, 0], [sin_2, 0, cos_2]])
        .double()
        .to(device),
        xyz.double().T,
    ).T

    tangent = leftwards[i][2] / ((leftwards[i][0] ** 2 + leftwards[i][1] ** 2) ** (0.5))
    cross_angle = math.atan(tangent)
    cos_3 = math.cos(cross_angle)
    sin_3 = math.sin(cross_angle)

    xyz = torch.matmul(
        torch.tensor([[1, 0, 0], [0, cos_3, -sin_3], [0, sin_3, cos_3]])
        .double()
        .to(device),
        xyz.double().T,
    ).T

    # Sensor at 1.2 meter above
    xyz[:, 2] -= 1200
    xyz = xyz.cpu().numpy()

    with h5py.File(
        f"./examples/vista_traces/lidar_{args.process}/lidar_3d.h5", "w"
    ) as f:
        f["timestamp"] = [[0], [0.1], [0.2]]
        f["xyz"] = [xyz]
        f["intensity"] = [las.intensity[indices]]

    f2 = h5py.File(f"./examples/vista_traces/lidar_{args.process}/lidar_3d.h5", "r")
    print(f2["timestamp"])
    print(f2["xyz"])
    print(f2["intensity"])

    pd.DataFrame(xyz).to_csv(
        f"./examples/vista_traces/lidar_{args.process}/lidar_3d.csv"
    )


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to .las file to convert to .h5")
    parser.add_argument("--frame", type=int, help="Frame number")
    parser.add_argument("--range", type=int, help="Range distance in metres")
    parser.add_argument("--process", type=int, help="Process number")
    args = parser.parse_args()

    main(args)
