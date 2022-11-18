import h5py
import argparse
import laspy
import numpy as np
import zipfile
import math
import csv
import random
from statistics import mean


def main(args):
    if args.input.endswith(".zip"):
        with zipfile.ZipFile(args.input, "r") as zip_ref:
            zip_ref.extractall("/tmp/lidar")

        args.input = f"/tmp/lidar/{args.input.split('/')[-1].split('.zip')[0]}"

        print(f"/tmp/lidar/{args.input.split('/')[-1].split('.zip')[0]}")

    las = laspy.read(args.input)

    trajectory = np.sort(
        las.points[
            np.array(
                np.where(
                    (las._points.scan_angle_rank == 0)
                    & (las._points.point_source_id == 2)
                )
            )
        ].array,
        order="gps_time",
    )[0]

    frame = random.randint(10000, len(trajectory) - 10000)
    try:
        pov = trajectory[frame]
    except IndexError:
        exit(1)

    last = trajectory[-1]
    if (
        (pov["X"] - last["X"]) ** 2
        + (pov["Y"] - last["Y"]) ** 2
        + (pov["Z"] - last["Z"]) ** 2
    ) ** (1 / 2) < 300000:
        exit(1)

    pov_X = pov["X"]
    pov_Y = pov["Y"]
    pov_Z = pov["Z"]
    pov_X_delta = []
    pov_Y_delta = []
    pov_Z_delta = []
    for i in range(1, 1000, 10):
        pov_next = trajectory[frame + i]
        pov_X_delta.append(pov_next["X"] - pov_X)
        pov_Y_delta.append(pov_next["Y"] - pov_Y)
        pov_Z_delta.append(pov_next["Z"] - pov_Z)

    pov_X_delta = mean(pov_X_delta)
    pov_Y_delta = mean(pov_Y_delta)
    pov_Z_delta = mean(pov_Z_delta)

    print((pov_X_delta**2 + pov_Y_delta**2 + pov_Z_delta**2) ** 0.5)

    x = np.array(las.X)
    y = np.array(las.Y)
    z = np.array(las.Z)

    xyz = np.array([x, y, z]).T

    # Traslantion
    xyz -= np.array([pov_X, pov_Y, pov_Z])

    # Rotation 1
    cos_1 = pov_X_delta / ((pov_X_delta**2 + pov_Y_delta**2) ** (0.5))
    sin_1 = pov_Y_delta / ((pov_X_delta**2 + pov_Y_delta**2) ** (0.5))
    xyz = np.dot(np.array([[cos_1, sin_1, 0], [-sin_1, cos_1, 0], [0, 0, 1]]), xyz.T).T

    # Rotation 2
    cos_2 = ((pov_X_delta**2 + pov_Y_delta**2) ** (0.5)) / (
        ((pov_X_delta**2 + pov_Y_delta**2) + pov_Z_delta**2) ** (0.5)
    )
    sin_2 = pov_Z_delta / (
        ((pov_X_delta**2 + pov_Y_delta**2) + pov_Z_delta**2) ** (0.5)
    )
    xyz = np.dot(np.array([[cos_2, 0, -sin_2], [0, 1, 0], [sin_2, 0, cos_2]]), xyz.T).T

    xyz_distance = np.sqrt(
        np.square(xyz[:, 0]) + np.square(xyz[:, 1]) + np.square(xyz[:, 2])
    )

    # Cross section angle
    cross_section = xyz[
        (xyz[:, 0] < 100)
        & (xyz[:, 0] > -100)
        & (xyz[:, 1] < 3000)
        & (xyz[:, 0] > -3000)
    ]
    tan_li = []
    for _ in range(100):
        random_idx = random.randint(0, cross_section.shape[0]-1)
        tan_li.append(cross_section[random_idx][2] / cross_section[random_idx][1])

    tangent = mean(tan_li)
    cross_angle = math.atan(tangent)
    cos_3 = math.cos(cross_angle)
    sin_3 = math.sin(cross_angle)

    xyz = np.dot(np.array([[1, 0, 0], [0, cos_3, -sin_3], [0, sin_3, cos_3]]), xyz.T).T

    # Sensor at 2 meter above
    xyz[:, 2] -= 2000

    indices = np.where((xyz_distance < 245000))
    xyz = xyz[indices]

    with h5py.File("./examples/vista_traces/lidar/lidar_3d.h5", "w") as f:
        f["timestamp"] = [[0], [0.1], [0.2]]
        f["xyz"] = [xyz]
        f["intensity"] = [las.intensity[indices]]

    f2 = h5py.File("./examples/vista_traces/lidar/lidar_3d.h5", "r")
    print(f2["timestamp"])
    print(f2["xyz"])
    print(f2["intensity"])

    # Cartesian voxelization
    aoi = xyz[np.where((xyz[:, 0] > 95000))]
    positives = aoi[np.random.choice(aoi.shape[0], 512, replace=False)]
    if len(positives) < 512:
        print("Too little amount of samples")
        exit(1)

    negatives = np.random.rand(10000, 3)
    negatives[:, 0] = negatives[:, 0] * 150000 + 95000
    negatives[:, 1] = (negatives[:, 1] - 0.5) * 150000
    negatives[:, 2] = (negatives[:, 2] - 0.5) * 30000
    discrete_aoi = np.unique(aoi.round(-3), axis=0)
    negatives = negatives[
        [
            i
            for i, v in enumerate(np.unique(negatives.round(-3), axis=0))
            if any(np.equal(discrete_aoi, v).all(1))
        ]
    ]
    print(negatives.shape)
    try:
        negatives = negatives[np.random.choice(negatives.shape[0], 512, replace=False)]
    except ValueError:
        print("Too little amount of samples")
        exit(1)

    with open("/tmp/lidar/trajectory.csv", "a") as f:
        f.write(f"{pov_X}, {pov_Y}, {pov_Z}, {sin_1}, {cos_1}, {sin_2}, {cos_2}\n")
    with open("/tmp/lidar/trajectory.csv", "r") as f:
        trajectory_info = list(csv.reader(f))

    positives = np.c_[positives, np.ones(positives.shape[0])]
    negatives = np.c_[negatives, np.zeros(negatives.shape[0])]
    positives[:, 0:3] /= 245000
    negatives[:, 0:3] /= 245000

    np.random.shuffle(positives)
    np.random.shuffle(negatives)

    gt_xyz = np.append(positives, negatives, axis=0)
    print(f"Number of samples: {gt_xyz.shape[0]}")

    np.savetxt(
        f"/home/sangwon/Desktop/lidar/{len(trajectory_info)}_gt.txt",
        gt_xyz,
        delimiter=",",
        fmt="%f",
    )

    with open("./examples/vista_traces/lidar/log.txt", "a") as f:
        f.write(
            f"{args.input.split('/')[-1].split('.zip')[0]}, {positives[0][0]}, {positives[0][1]}, {positives[0][2]}, {negatives[0][0]}, {negatives[0][1]}, {negatives[0][2]}\n"
        )


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to .las file to convert to .h5")
    # parser.add_argument("--frame", type=int, help="Frame number")
    args = parser.parse_args()

    main(args)
