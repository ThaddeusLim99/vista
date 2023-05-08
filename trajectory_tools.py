import numpy as np
import argparse
import tkinter as tk
import tkinter.filedialog
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

def parse_cmdline_args() -> argparse.Namespace:
  # use argparse to parse arguments from the command line
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--config", type=str, default=None, help="Path to sensor config file")
  parser.add_argument("--trajectory", type=str, default=None, help="Path to trajectory folder")
  parser.add_argument("--observer_height", type=float, default=1.8, help="Height of the observer in m")
  parser.add_argument("--scenes", type=str, default=None, help="Path to the Vista output folder")
  
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
    if args.trajectory == None:
        # Manually open trajectory folder
        Tk().withdraw()
        trajectory_folderpath = tk.filedialog.askdirectory(
            initialdir=ROOT2, title="Please select the trajectory folder"
        )
        print(f"You have chosen to open the trajectory folder:\n{trajectory_folderpath}")

    else:
        # Use trajectory folder from defined command line argument
        trajectory_folderpath = args.trajectory
        print(f"You have chosen to use the pregenerated trajectory folder:\n{trajectory_folderpath}")

    # Read the filenames of the trajectories into a list
    trajectory_files = [
        path
        for path in os.listdir(trajectory_folderpath)
        if os.path.isfile(os.path.join(trajectory_folderpath, path))
    ]
    
    # Sanity check
    #if len(trajectory_files) != 5:
    #  raise(RuntimeError(f"Trajectory folder is missing files!\nExpected count: 5 (got {len(trajectory_files)})!"))
    assert len(trajectory_files) == 5, f"Trajectory folder is missing files!\nExpected count: 5 (got {len(trajectory_files)})!"

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
    assert observer_points.shape == road_points.shape == forwards.shape == leftwards.shape == upwards.shape, f"Bad trajectory files! One or more trajectories are missing points!"
    
    # Correct the z-component of our forward vector
    useCorrectedZ = True
    if useCorrectedZ:
        print(f"Using the corrected z-compoment of the forward vector!")
        
        forwards[:][2] = (
            -(upwards[:][0] * forwards[:][0] + upwards[:][1] * forwards[:][1])
            / upwards[:][2]
        )
        magnitude = (forwards[:][0] ** 2 + forwards[:][1] ** 2 + forwards[:][2] ** 2) ** (
            1 / 2
        )
        forwards /= magnitude    
       
    # Finally store the trajectory values into our object
    trajectory = Trajectory(
        observer_points=observer_points,
        road_points=road_points,
        forwards=forwards,
        leftwards=leftwards,
        upwards=upwards,
    )
    
    print(f"{road_points.shape[0]} trajectory points have been loaded for the corresponding trajectory folder {os.path.basename(trajectory_folderpath)}.")

    return trajectory

def obtain_scene_path(args: argparse.Namespace) -> str:
    
    """Obtains the path to the folder containing all of the outputs
    to the Vista simulator.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        scenes_folderpath (str): Path to the folder containing the Vista outputs.
    """

    # Get trajectory folder path
    if args.scenes == None:
        # Manually open trajectory folder
        Tk().withdraw()
        scenes_folderpath = tk.filedialog.askdirectory(
            initialdir=ROOT2, title="Please select the Vista output folder"
        )
        print(f"\nYou have chosen to open the folder to the scenes:\n{scenes_folderpath}")

    else:
        # Use trajectory folder from defined command line argument
        scenes_folderpath = args.scenes
        print(f"\nYou have chosen to use the predefined path to the scenes:\n{scenes_folderpath}")

    num_scenes = len([name for name in os.listdir(scenes_folderpath) if os.path.isfile(os.path.join(scenes_folderpath, name))])
    print(f"{num_scenes} scenes were found for the corresponding road section folder {os.path.basename(scenes_folderpath)}")

    return scenes_folderpath