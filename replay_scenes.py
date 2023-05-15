import numpy as np
import open3d as o3d
import pandas
import argparse
import tkinter as tk
import sys, os
import glob
import time
from tkinter import Tk
from pathlib import Path


from trajectory_tools import parse_cmdline_args, obtain_scene_path

# Global variables for file I/O
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
ROOT2 = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
  sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def open_point_cloud(path2scenes: str, frame: int, res: np.float32, ext: str = ".txt"):
  # outputs are in the format output_FRAME_0.11.txt
  # split the first scene file to obtain the sensor resolution...

  scene_name = f"output_{frame}_{res:.2f}{ext}"
  print(f"Loaded scene {scene_name}")
  
  path_to_scene = os.path.join(path2scenes, scene_name)
  
  # We don't want spherical coordinates in our scene data
  xyzypd = np.genfromtxt(path_to_scene, delimiter=",")
  xyz = np.delete(xyzypd, [3,4,5], axis=1)
  
  
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyz)

  vis = o3d.visualization.Visualizer()
  vis.create_window()
  vis.add_geometry(pcd)
  time.sleep(5)
  ctr = vis.get_view_control()
  ctr.set_lookat([0,0,0])
  ctr.set_zoom(0.5)
  vis.run()
  vis.destroy_window()

  return pcd

def replay_scenes(path2scenes: str, scenes_list: list, res: np.float32, ext: str = "*.txt") -> None:
  print(f"Visualizing the scenes given by path {path2scenes}")
  # Example taken from open3d non-blocking visualization...
  # vis = o3d.visualization.Visualizer()
  # vis.create_window()
  
  # geometry = o3d.geometry.PointCloud()
  geometry = open_point_cloud(path2scenes, 0, res, ".txt")
  # print(type(geometry))
  # vis.add_geometry(geometry)
  #v#is.update_geometry(geometry)
  #vis.poll_events()
  #vis.update_renderer()
  #vis.run()
  #vis.destroy_window()  
  '''
  for frame in range(1, len(scenes_list)):
    geometry.points = open_point_cloud(path2scenes, frame, res, ".txt").points
    

    vis.update_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()
  '''

  #vis.destroy_window()
  return

def obtain_scenes_list_details(path2scenes: str, ext: str = "*.txt") -> list or np.float32:
  
  path2scenes = os.path.join(path2scenes, ext)
  
  # get list of filenames within our scenes list
  # filenames are guaranteed to be of the format "output_FRAME_SENSORRES.txt"
  filenames = [os.path.basename(abs_path) for abs_path in glob.glob(path2scenes)]
  
  res = np.float32(
        float(os.path.splitext(
            (filenames[0].split("_")[-1])   
        )[0])
      )

  return filenames, res

def main():
  args = parse_cmdline_args()
  path_to_scenes = obtain_scene_path(args)
  scenes_list, res = obtain_scenes_list_details(path_to_scenes)
  replay_scenes(path_to_scenes, scenes_list, res, "*.txt")


  # open_point_cloud(path_to_scenes)
  
  return


if __name__ == "__main__":
  main()