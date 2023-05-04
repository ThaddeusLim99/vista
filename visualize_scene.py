import utm
import numpy as np
import argparse
import sys
import os
import tkinter as tk
import tkinter.filedialog
from tkinter import Tk
from pathlib import Path
import webbrowser

'''
Scene visualization
Eric Cheng
2023-05-04
Given a Vista output scene, visualize this scene from the vehicle's 
rough POV in Google Maps.
'''

# Global variables for file I/O
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
ROOT2 = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
  sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

class Trajectory:
  """Container class for the trajectory. Copy pasted and slightly modified from sensorpoints.py
  """
  def __init__(
    self, 
    observer_points: np.ndarray, 
    road_points: np.ndarray, 
    forwards: np.ndarray, 
    leftwards: np.ndarray,
    upwards: np.ndarray
    ) -> None:
    
    self.__observer_points = observer_points
    self.__road_points = road_points
    self.__forwards = forwards
    self.__leftwards = leftwards
    self.__upwards = upwards
    
    pass

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


'''
Global coordinates (from road_points.csv):
x (UTM Easting)
y (UTM Northing)

Obtain our heading using directional vectors? 
heading = (lon-centralmeridian) * sin(lat)
where centralmeridian_longitude = (6*UTMZone - 183) 
# note for alberta, utm zone 12
https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=53.581139%2C-113.549043&heading=-2&pitch=0&fov=80

googlemapangle = -((atan2(forwardvec[1], forwardvec[0])*(180/math.pi)-90) + convergence???????)
'''
# googlemapangle = -((atan2(forwardvec[1], forwardvec[0])*(180/math.pi)-90) + convergence)


# Copied and slightly modified from obtain_trajectory() in sensorpoints.py
def obtain_trajectory_details():

  # Manually open trajectory folder
  Tk().withdraw();
  trajectory_folderpath = tk.filedialog.askdirectory(
    initialdir = ROOT2,
    title = "Please select the trajectory folder"
  )
  print(f"You have chosen to open the trajectory folder:\n{trajectory_folderpath}")   

  
  trajectory_files = [
    path for path in os.listdir(trajectory_folderpath) 
    if os.path.isfile(os.path.join(trajectory_folderpath, path))
    ]

  # Read the csv files and store them into the function
  trajectory_data = dict()
  
  for csv in trajectory_files:
    csv_noext = os.path.splitext(csv)[0]
    path_to_csv = os.path.join(trajectory_folderpath, csv)
    data = np.genfromtxt(path_to_csv, delimiter=",")
    trajectory_data[csv_noext] = data

  observer_points = trajectory_data["observer_points"]
  road_points     = trajectory_data["road_points"]
  forwards        = trajectory_data["forwards"]
  leftwards       = trajectory_data["leftwards"]
  upwards         = trajectory_data["upwards"]

  
  # Finally store our trajectory values into our object
  trajectory = Trajectory(
    observer_points=observer_points,
    road_points=road_points,
    forwards=forwards,
    leftwards=leftwards,
    upwards=upwards
    )
  
  return trajectory

def visualize_scene(
  road_points: np.ndarray, 
  forward_vecs: np.ndarray,
  utm_zone: int, 
  scene: int) -> None:
  
  """
  Visualises the real world location of a particular Vista scene into Google Maps.
  """

  lat, lon = utm.to_latlon(road_points[scene, 0], road_points[scene, 1], utm_zone, 'U')

  # Central meridian of our given UTM zone in longitude
  central_meridian = (6*utm_zone - 183)
  # Angle of local UTM 'north' relative to true North
  convergence_angle = (lon-central_meridian)*np.sin(lat) 
  
  # To get the heading relative to true north:
  # Obtain the angle of the forward vector relative to our Cartesian coordinate system (in UTM)
  # Translate it by -90 degrees to convert our forward vector angle from CCW East to CCW North
  # Then add it with our convergence angle to make it relative to true North, which Google Maps uses.
  # Then apply a negative sign for correcting the direction.
  heading = -( ((np.arctan2(forward_vecs[scene, 1], forward_vecs[scene, 0])*(180/np.pi)) - 90) + convergence_angle)
  
  # open up the heading and latlon from google maps street view
  # just some settings for visualization
  pitch = 0
  fov = 120
  
  link=f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat}%2C{lon}&heading={heading}&pitch={pitch}&fov={fov}"
  webbrowser.open_new_tab(link)

  return

def parse_cmdline_args() -> argparse.Namespace:
  return

def main() -> None:
  # Temporary for now
  utm_zone = 12
  trajectory = obtain_trajectory_details()
  roadpoints = trajectory.getRoadPoints()
  forwardvecs = trajectory.getForwards()
  
  usrinput = None
  while(usrinput != 'q'):
    
    usrinput = input(f"Enter the scene number (0 - {roadpoints.shape[0]}) (type 'q' to quit): ")
    if usrinput == "q":
      break
    
    visualize_scene(roadpoints, forwardvecs, utm_zone, int(usrinput))
  
  
  # debug lol
  # rpts = np.genfromtxt("road_points.csv", delimiter=',')
  # fvecs = np.genfromtxt("forwards.csv", delimiter=',')
  # visualize_scene(rpts, fvecs, 12, scene=1529)
  return

if __name__ == "__main__":
  main()