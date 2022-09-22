import h5py
import argparse
import laspy
import numpy as np


def main(args):
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

    pov = trajectory[10000]
    pov_next = trajectory[151]
    pov_X = pov["X"]
    pov_Y = pov["Y"]
    pov_Z = pov["Z"] + 2000
    pov_X_delta = pov_next["X"] - pov_X
    pov_Y_delta = pov_next["Y"] - pov_Y
    pov_Z_delta = pov_next["Z"] - pov_Z + 2000
    print((pov_X_delta**2 + pov_Y_delta**2 + pov_Z_delta**2) ** 0.5)

    x = np.array([las.X])
    y = np.array([las.Y])
    z = np.array([las.Z])

    cos = pov_X_delta / ((pov_X_delta**2 + pov_Y_delta**2) ** (0.5))
    sin = pov_Y_delta / ((pov_X_delta**2 + pov_Y_delta**2) ** (0.5))
    [x, y] = np.dot(
        np.array([x - pov_X, y - pov_Y]).T, np.array([[cos, -sin], [sin, cos]])
    ).T

    cos = ((pov_X_delta**2 + pov_Y_delta**2) ** (0.5)) / (
        ((pov_X_delta**2 + pov_Y_delta**2) + pov_Z_delta**2) ** (0.5)
    )
    sin = pov_Z_delta / (
        ((pov_X_delta**2 + pov_Y_delta**2) + pov_Z_delta**2) ** (0.5)
    )
    z = np.dot(np.array([x, z - pov_Z]).T, np.array([[cos, -sin], [sin, cos]])).T[1]

    xyz = np.array([x, y, z]).T
    new_trajectory = xyz[
        np.array(
            np.where(
                (las._points.scan_angle_rank == 0) & (las._points.point_source_id == 2)
            )
        )
    ][0]

    np.savetxt(
        "./examples/vista_traces/lidar/input.csv",
        xyz.reshape([len(xyz), 3]),
        delimiter=",",
        fmt="%f",
    )
    np.savetxt(
        "./examples/vista_traces/lidar/trajectory.csv",
        new_trajectory.reshape([len(new_trajectory), 3]),
        delimiter=",",
        fmt="%f",
    )

    with h5py.File("/".join(args.input.split("/")[:-1]) + "/lidar_3d.h5", "w") as f:
        f["timestamp"] = [[0], [0.1], [0.2]]
        f["xyz"] = [xyz]
        f["intensity"] = [las.intensity]

    f2 = h5py.File("/".join(args.input.split("/")[:-1]) + "/lidar_3d.h5", "r")
    print(f2["timestamp"])
    print(f2["xyz"])
    print(f2["intensity"])


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to .las file to convert to .h5")
    args = parser.parse_args()

    main(args)
