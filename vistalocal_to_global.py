import numpy as np
import pandas as pd
import torch
import sys
import os
import math
import argparse
import torch.multiprocessing as mp  # Experimental
from tqdm import tqdm
from pathlib import Path

import trajectory_tools


"""
Actual values of global coordinates (points given in the road section) are given in UTM coordinates, in meters.

pov_X, pov_Y, pov_Z - road points, in x, y, z coordinates repsectively.
xyz: global coordinates, in mm
~~~
Transform our points at frame i, given in GLOBAL coordinates to LOCAL coordinates,
through the following steps:

### inverse of translation: add the road points instead of subtracting them ###
xyz -= np.array([pov_X, pov_Y, pov_Z])

### here the numpy array containing our points is converted into a CUDA tensor

# Rotation 1
cos_1 = forwards[i][0] / ((forwards[i][0] ** 2 + forwards[i][1] ** 2) ** (0.5))
sin_1 = forwards[i][1] / ((forwards[i][0] ** 2 + forwards[i][1] ** 2) ** (0.5))
### inverse of rotation 1: cos -sin sin cos ###
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
### inverse of rotation 2: cos sin -sin cos ###
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

### inverse of rotation 3: <INVERSE_OF_MATRIX> ###
xyz = torch.matmul(
    torch.tensor([[1, 0, 0], [0, cos_3, -sin_3], [0, sin_3, cos_3]])
    .double()
    .to(device),
    xyz.double().T,
).T

# Sensor at 1.2 meter above
# inverse of our observer point translation: add 1800 to each of them
xyz[:, 2] -= 1800

##### 

In order to reverse the transformations, we will have to look at the XYZ points in each Vista scene,
and then apply inverse of the translations mentioned above, in reverse order.



"""

# Global variables for file I/O
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
ROOT2 = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


def transform_scene(
    frame: int,
    path: str,
    trajectory: trajectory_tools.Trajectory,
    device: str,
    sensor_res: float = 0.11,
) -> None:
    """Transforms an individual Vista scene from local coordinates (in mm) to
    global coordinates (in UTM, meters)

    Outputs will be written as so.

    Args:
        frame (int): The frame of the specific Vista scene.
        path (str): The path to the folder containing the Vista scenes.
        trajectory (trajectory_tools.Trajectory): The container class for our respective
        trajectory.
        device (str): The device at which we will handle tensors with
        (cuda:0 if we are using CUDA), (cpu:0 if we are using CPU)
        sensor_res (float): The resolution of the sensor. Hardcoded into the function
        defnition for now, need to parse from the sensor configuration later.

    """
    # Transform our scene from local coordinates back to global coordinates

    ## Order to convert from global to local:
    #
    # Convert our road points in m into mm by multiplying by 1000
    # Convert our x,y,z in m to mm by multiplying by 1000
    # Translate all of our x,y,z into local by subtracting the road_points in mm
    #
    # Cull all xyz points that are above the sensor range (not needed here)
    # CONVERT OUR XYZ FROM NUMPY ARRAY INTO A CUDA TENSOR (torch.Tensor)
    # (xyz=torch.tensor(xyz).to(device))
    #
    # Rotation 1: Multiply by corresponding transformation matrix (given)
    # Rotation 2: Same as above
    # Rotation 3: Same as above
    # Translate sensor down by observer height in mm
    # CONVERT OUR XYZ FROM A CUDA TENSOR INTO NUMPY ARRAY
    # (xyz=xyz.cpu().numpy())

    def open_vista_scene(frame: int, path: str) -> np.ndarray:
        """Opens a singular Vista scene as a numpy array.
        Each scene is given in XYZ points, and in mm.

        Args:
            frame (int): The number of the scene that we want to open.
            path (str): Path to the Vista output folder.

        Returns:
            scene (np.ndarray): The points that make up the scene at a specific
            frame, given in XYZ local (in mm).
        """

        filename = f"output_{frame}_{sensor_res}.txt"
        path = os.path.join(path, filename)
        scene = pd.read_csv(path, skiprows=0).to_numpy()  # Skip the header row

        # Remove the corresponding spherical coordinates in our Vista output
        scene = np.delete(scene, [3, 4, 5], axis=1)

        return scene

    # Prepare our points for the transformation
    xyz = open_vista_scene(frame, path)

    # Prepare our trajectory data for the transformation
    traj = trajectory.getRoadPoints()
    pov_X = (traj[frame][0]) * 1000
    pov_Y = (traj[frame][1]) * 1000
    pov_Z = (traj[frame][2]) * 1000

    forwards = trajectory.getForwards()
    leftwards = trajectory.getLeftwards()
    upwards = trajectory.getUpwards()
    
    # Fix the z component of the forwards vector
    forwards[frame][2] = (
        -(upwards[frame][0] * forwards[frame][0] + upwards[frame][1] * forwards[frame][1])
        / upwards[frame][2]
    )
    magnitude = (forwards[frame][0] ** 2 + forwards[frame][1] ** 2 + forwards[frame][2] ** 2) ** (
        1 / 2
    )
    forwards /= magnitude

    ### INVERSE OF THE CONVERT_SINGLE PROCESS BEGINS HERE ###

    # Convert our scene from a numpy array to a CUDA tensor
    xyz = torch.tensor(xyz).to(device)

    # Undo observer height translation
    xyz[:, 2] += 1800

    # Undo rotation 3 (along x-axis, to CCW)
    tangent = leftwards[frame][2] / (
        (leftwards[frame][0] ** 2 + leftwards[frame][1] ** 2) ** (0.5)
    )
    cross_angle = math.atan(tangent)
    cos_3 = math.cos(cross_angle)
    sin_3 = math.sin(cross_angle)

    xyz = torch.matmul(
        torch.tensor([[1, 0, 0], [0, cos_3, sin_3], [0, -sin_3, cos_3]])
        .double()
        .to(device),
        xyz.double().T,
    ).T

    # Undo rotation 2 (along y-axis, to CW)
    cos_2 = ((forwards[frame][0] ** 2 + forwards[frame][1] ** 2) ** (0.5)) / (
        ((forwards[frame][0] ** 2 + forwards[frame][1] ** 2) + forwards[frame][2] ** 2)
        ** (0.5)
    )
    
    sin_2 = forwards[frame][2] / (
        ((forwards[frame][0] ** 2 + forwards[frame][1] ** 2) + forwards[frame][2] ** 2)
        ** (0.5)
    )

    xyz = torch.matmul(
        torch.tensor([[cos_2, 0, sin_2], [0, 1, 0], [-sin_2, 0, cos_2]])
        .double()
        .to(device),
        xyz.double().T,
    ).T

    # Undo rotation 1 (along z-axis, to CW)
    cos_1 = forwards[frame][0] / (
        (forwards[frame][0] ** 2 + forwards[frame][1] ** 2) ** (0.5)
    )
    sin_1 = forwards[frame][1] / (
        (forwards[frame][0] ** 2 + forwards[frame][1] ** 2) ** (0.5)
    )

    xyz = torch.matmul(
        torch.tensor([[cos_1, -sin_1, 0], [sin_1, cos_1, 0], [0, 0, 1]])
        .double()
        .to(device),
        xyz.double().T,
    ).T

    # Undo conversion of our xyz points from a numpy array to a CUDA tensor
    xyz = xyz.cpu().numpy()

    # Undo translation
    xyz += np.array([pov_X, pov_Y, pov_Z])

    # Undo conversion from m to mm (convert our xyz back to meters)
    xyz /= 1000


    # Now that we have our XYZ coordinates in global (m),
    # we can now write the output to a csv file.
    df = pd.DataFrame(xyz)
    df.columns = ["x", "y", "z"]

    # Get our path to the output folder
    outpath = f"{ROOT2}/examples/vista_traces/lidar_output/{os.path.basename(path)}_global"
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Get name of output file
    filename = f"output_{frame}_{sensor_res:.2f}_global.csv"
    outpath_file = os.path.join(outpath, filename)

    df.to_csv(outpath_file, index=False)


    return 


def main() -> None:
    traj_args = trajectory_tools.parse_cmdline_args()
    trajectory = trajectory_tools.obtain_trajectory_details(traj_args)
    path_to_scenes = trajectory_tools.obtain_scene_path(traj_args)

    # Sanity check: Make sure that the number of Vista scenes equal the trajectory datapoints
    num_traj_points = trajectory.getRoadPoints().shape[0]
    num_scenes = len(
        [
            name
            for name in os.listdir(path_to_scenes)
            if os.path.isfile(os.path.join(path_to_scenes, name))
        ]
    )

    assert (
        num_traj_points == num_scenes
    ), f"The number of trajectory data points does not equal the number of Vista scenes!"

    device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
    print(f"\nUsing device {device} for the conversion...")

    # transform_scene(frame=900, path=path_to_scenes, trajectory=trajectory, device=device, sensor_res=0.11)
    
    
    # Loop to convert for each frame goes here
    # Could be parallelized?
    useParallel = True
    sensor_res = 0.11

    if useParallel:
        mp.freeze_support()
        cores = mp.cpu_count() - 1

        try:
            mp.set_start_method("spawn")  # For CUDA; tensors are placed onto GPU memory
        except RuntimeError:
            pass

        # Parallelization thing, we are going to have to see if it works...?
        # Hopefully using CUDA with a parallel for loop won't freeze anythong
        print(
            f"Converting {num_scenes} scenes using parallel pool with {cores} cores..."
        )
        with mp.Pool(cores) as p:  # Opening up more process pools
            scenes_list = p.starmap(
                transform_scene,
                tqdm(
                    [
                        (frame, path_to_scenes, trajectory, device, sensor_res)
                        for frame in range(num_scenes)
                    ],
                    total=num_scenes,
                ),
            )
            p.close()  # No new tasks for our pool
            p.join()  # Wait for all processes to finish

    else:
        # Very slow???
        
        for frame in tqdm(range(num_scenes)):
            transform_scene(
                frame=frame,
                path=path_to_scenes,
                trajectory=trajectory,
                device=device,
                sensor_res=sensor_res,
            )

    # Get our path to the output folder
    outpath = f"{ROOT2}/examples/vista_traces/lidar_output/{os.path.basename(path_to_scenes)}_global"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    for scene in tqdm(scenes_list):
        # Now that we have our XYZ coordinates in global (m),
        # we can now write the output to a csv file.
        df = pd.DataFrame(scene)
        df.columns = ["x", "y", "z"]

        # Get name of output file
        filename = f"output_{frame}_{sensor_res:.2f}_global.csv"
        outpath_file = os.path.join(outpath, filename)

        df.to_csv(outpath_file, index=False)
        
    # Another sanity check
    num_converted_scenes = len(
        [
            name
            for name in os.listdir(outpath)
            if os.path.isfile(os.path.join(outpath, name))
        ]
    )

    print(
        f"Processing complete.\n{num_converted_scenes} scenes were converted to global coordinates and are written to \n{outpath}"
    )
    
    return


if __name__ == "__main__":
    main()
