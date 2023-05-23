import numpy as np
import argparse
import tkinter as tk
import tkinter.filedialog
import laspy
import sys
import os
from tkinter import Tk
from pathlib import Path

"""
Tools for obtaining an already generated trajectory.
"""

# TODO For visualize_scene.py, sensorpoints.py, vistalocal_to_global.py,
# use this instead of manually coding everything for each file...

# Global variables for file I/O
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
ROOT2 = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


class Trajectory:
    """Container class for the trajectory"""

    def __init__(
        self,
        observer_points: np.ndarray,
        road_points: np.ndarray,
        forwards: np.ndarray,
        leftwards: np.ndarray,
        upwards: np.ndarray,
    ) -> None:
        self.__observer_points = observer_points
        self.__road_points = road_points
        self.__forwards = forwards
        self.__leftwards = leftwards
        self.__upwards = upwards

        pass

    # Getters
    def getObserverPoints(self) -> np.ndarray:
        return self.__observer_points

    def getRoadPoints(self) -> np.ndarray:
        return self.__road_points

    def getForwards(self) -> np.ndarray:
        return self.__forwards

    def getLeftwards(self) -> np.ndarray:
        return self.__leftwards

    def getUpwards(self) -> np.ndarray:
        return self.__upwards

    def getNumPoints(self) -> np.int32:
        return self.__road_points.shape[0]

    # Setters (just in case if we want to work with future trajectories)
    def setObserverPoints(self, observer_points: np.ndarray) -> None:
        self.__observer_points = observer_points

    def setRoadPoints(self, road_points: np.ndarray) -> None:
        self.__road_points = road_points

    def setForwards(self, forwards: np.ndarray) -> None:
        self.__forwards = forwards

    def setLeftwards(self, leftwards: np.ndarray) -> None:
        self.__leftwards = leftwards

    def setUpwards(self, upwards: np.ndarray) -> None:
        self.__upwards = upwards

class LasPointCloud:
    """
    Container class for the .las file. Cuts down on unused fields from the
    raw .las file itself.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        gps_time: np.ndarray,
        scan_angle_rank: np.ndarray,
        point_source_ID: np.ndarray,
        intensity: np.ndarray,
        lasfilename: str,
    ):
        self.__x = x
        self.__y = y
        self.__z = z
        self.__gps_time = gps_time
        self.__scan_angle_rank = scan_angle_rank
        self.__point_source_ID = point_source_ID
        self.__intensity = intensity
        self.__lasfilename = lasfilename

    pass
    
    # Getters 
    def getX(self) -> np.ndarray:
        return self.__x

    def getY(self) -> np.ndarray:
        return self.__y

    def getZ(self) -> np.ndarray:
        return self.__z

    def getGPSTime(self) -> np.ndarray:
        return self.__gps_time

    def getScanAngleRank(self) -> np.ndarray:
        return self.__scan_angle_rank

    def getPointSourceID(self) -> np.ndarray:
        return self.__point_source_ID
    
    def getIntensity(self) -> np.ndarray:
        return self.__intensity

    def getLasFileName(self) -> str:
        return self.__lasfilename
    
    # Setters (just in case if we want to work with future .las clouds, but these probably shouldn't be used)
    def setX(self, x: np.ndarray) -> None:
        self.__x = x

    def setY(self, y: np.ndarray) -> None:
        self.__y = y
        
    def setZ(self, z: np.ndarray) -> None:
        self.__z = z

    def setGPSTime(self, gps_time: np.ndarray) -> None:
        self.__gps_time = gps_time
        
    def setScanAngleRank(self, scan_angle_rank: np.ndarray) -> None:
        self.__scan_angle_rank = scan_angle_rank
        
    def setPointSourceID(self, point_source_id: np.ndarray) -> None:
        self.__point_source_ID = point_source_id
        
    def setIntensity(self, intensity: np.ndarray) -> None:
        self.__intensity = intensity
        
    def setLasFileName(self, lasfilename: str) -> None:
        self.__lasfilename = lasfilename

    pass


def parse_cmdline_args() -> argparse.Namespace:
    # use argparse to parse arguments from the command line
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=str, default=None, help="Path to sensor config file"
    )
    parser.add_argument(
        "--trajectory", type=str, default=None, help="Path to trajectory folder"
    )
    parser.add_argument(
        "--observer_height", type=float, default=1.8, help="Height of the observer in m"
    )
    parser.add_argument(
        "--scenes", type=str, default=None, help="Path to the Vista output folder"
    )
    parser.add_argument(
        "--input", type=str, default=None, help="Path to the .las file"
    )

    return parser.parse_args()

def obtain_trajectory_details(args: argparse.Namespace) -> Trajectory:
    """Obtains a pregenerated trajectory and reads each of them into
    a container class.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Trajectory: Container class for our imported trajectory data.
    """

    # Get trajectory folder path
    try:
        arg_trajectory = args.trajectory
    except AttributeError:
        arg_trajectory = None

    if arg_trajectory == None:
        # Manually open trajectory folder
        Tk().withdraw()
        trajectory_folderpath = tk.filedialog.askdirectory(
            initialdir=ROOT2, title="Please select the trajectory folder"
        )
        print(
            f"You have chosen to open the trajectory folder:\n{trajectory_folderpath}"
        )

    else:
        # Use trajectory folder from defined command line argument
        trajectory_folderpath = args.trajectory
        print(
            f"You have chosen to use the pregenerated trajectory folder:\n{trajectory_folderpath}"
        )

    # Read the filenames of the trajectories into a list
    trajectory_files = [
        path
        for path in os.listdir(trajectory_folderpath)
        if os.path.isfile(os.path.join(trajectory_folderpath, path))
    ]

    # Sanity check
    # if len(trajectory_files) != 5:
    #  raise(RuntimeError(f"Trajectory folder is missing files!\nExpected count: 5 (got {len(trajectory_files)})!"))
    assert (
        len(trajectory_files) == 5
    ), f"Trajectory folder is missing files!\nExpected count: 5 (got {len(trajectory_files)})!"

    # Read each of the csv files as numpy arrays
    trajectory_data = dict()

    for csv in trajectory_files:
        csv_noext = os.path.splitext(csv)[0]
        path_to_csv = os.path.join(trajectory_folderpath, csv)
        data = np.genfromtxt(path_to_csv, delimiter=",")
        trajectory_data[csv_noext] = data

    observer_points = trajectory_data["observer_points"]
    road_points = trajectory_data["road_points"]
    forwards = trajectory_data["forwards"]
    leftwards = trajectory_data["leftwards"]
    upwards = trajectory_data["upwards"]

    # Another sanity check
    assert (
        observer_points.shape
        == road_points.shape
        == forwards.shape
        == leftwards.shape
        == upwards.shape
    ), f"Bad trajectory files! One or more trajectories are missing points!"

    # Correct the z-component of our forward vector FIXME This is broken, fix later...
    useCorrectedZ = True
    if useCorrectedZ:
        print(f"Using the corrected z-compoment of the forward vector!")
        forwards[:, 2] = (
            -(upwards[:, 0] * forwards[:, 0] + upwards[:, 1] * forwards[:, 1])
            / upwards[:, 2]
        )

        magnitude = (
            forwards[:, 0] ** 2 + forwards[:, 1] ** 2 + forwards[:, 2] ** 2
        ) ** (1 / 2)

        forwards[:, 2] /= magnitude

    # Finally store the trajectory values into our object
    trajectory = Trajectory(
        observer_points=observer_points,
        road_points=road_points,
        forwards=forwards,
        leftwards=leftwards,
        upwards=upwards,
    )

    print(
        f"{road_points.shape[0]} trajectory points have been loaded for the corresponding trajectory folder {os.path.basename(trajectory_folderpath)}"
    )

    return trajectory


def obtain_scene_path(args: argparse.Namespace) -> str:
    """Obtains the path to the folder containing all of the outputs
    to the Vista simulator.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        scenes_folderpath (str): Path to the folder containing the Vista outputs.
    """
    try:
        arg_scene = args.scenes
    except AttributeError:
        arg_scene = None

    # Get trajectory folder path
    if arg_scene == None:
        # Manually open trajectory folder
        Tk().withdraw()
        scenes_folderpath = tk.filedialog.askdirectory(
            initialdir=ROOT2, title="Please select the Vista output folder"
        )
        print(
            f"\nYou have chosen to open the folder to the scenes:\n{scenes_folderpath}"
        )

    else:
        # Use trajectory folder from defined command line argument
        scenes_folderpath = args.scenes
        print(
            f"\nYou have chosen to use the predefined path to the scenes:\n{scenes_folderpath}"
        )

    num_scenes = len(
        [
            name
            for name in os.listdir(scenes_folderpath)
            if os.path.isfile(os.path.join(scenes_folderpath, name))
        ]
    )
    print(
        f"\n{num_scenes} scenes were found for the corresponding road section folder."
    )

    return scenes_folderpath


def open_las(args: argparse.Namespace):
    """
    Opens a .las file when prompted to do so. Can force a predetermined filename
    (default called as None for manual input)

    Arguments:
    verbose (bool): Setting to print extra information to the command line.

    predetermined_filename (string): The predetermined file name of the point cloud.
    User can be manually prompted to enter the point cloud, or it can be set to some
    point cloud via command line for automation. See main() for command line syntax.
    """
    try:
        arg_input = args.input
    except AttributeError:
        arg_input = None
    
    if arg_input == None:
        # Manually obtain file via UI
        Tk().withdraw()
        las_filename = tk.filedialog.askopenfilename(
            filetypes=[(".las files", "*.las"), ("All files", "*")],
            initialdir="inputs/",
            title="Please select the main point cloud",
        )

        print(f"You have chosen to open the point cloud:\n{las_filename}")

    else:
        las_filename = args.input

    # Obtain the las file name itself rather than the path for csv output
    las_filename_cut = os.path.basename(las_filename)

    # Note: lowercase dimensions with laspy give the scaled value
    raw_las = laspy.read(las_filename)
    las = LasPointCloud(
        raw_las.x,
        raw_las.y,
        raw_las.z,
        raw_las.gps_time,
        raw_las.scan_angle_rank,
        raw_las.point_source_id,
        raw_las.intensity,
        las_filename_cut,
    )

    return las
