import numpy as np
import multiprocessing as mp
import open3d as o3d
import pandas
import argparse
import tkinter as tk
import sys, os
import glob
from tqdm import tqdm
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

class _PointCloudPickleable:
  def __init__(self, points = np.ndarray):
    self.points = points
    pass
  
  def create_point_cloud(self):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(self.points)
    return pcd


def open_point_cloud(path2scenes: str, frame: int, res: np.float32) -> _PointCloudPickleable:
  # outputs are in the format output_FRAME_0.11.txt
  # split the first scene file to obtain the sensor resolution...

  scene_name = f"output_{frame}_{res:.2f}.txt"
  path_to_scene = os.path.join(path2scenes, scene_name)

  xyzypd = np.genfromtxt(path_to_scene, delimiter=",")
  xyz = np.delete(xyzypd, [3,4,5], axis=1) # We don't want spherical coordinates in our scene data
  xyz = np.delete(xyz, 0, axis=0)          # We don't want our header either since it reads into nan
  xyz /= 1000   # Convert from mm to m
  
  # pcd = o3d.geometry.PointCloud()
  # pcd.points = o3d.utility.Vector3dVector(xyz)
  return _PointCloudPickleable(xyz)

def replay_scenes(path2scenes: str, scenes_list: np.ndarray, res: np.float32, offset: int) -> None:
  
  print(f"Visualizing the scenes given by path {path2scenes}")
  # Example taken from open3d non-blocking visualization...
  vis = o3d.visualization.Visualizer()
  vis.create_window()
  
  # Render options
  render_opt = vis.get_render_option() 
  render_opt.point_size = 2.0
  render_opt.show_coordinate_frame = True
  # 8-bit RGB, (16, 16, 16)
  render_opt.background_color = np.array([16/255, 16/255, 16/255])
  
  '''
  geometry = open_point_cloud(path2scenes, 0+offset, res, ".txt")
  # print(type(geometry))
  vis.add_geometry(geometry)
  vis.update_geometry(geometry)
  #vis.poll_events()
  vis.update_renderer()
  vis.run()
  vis.destroy_window()  
  '''
  
  geometry = o3d.geometry.PointCloud()
  vis.add_geometry(geometry)

  for frame, scene in enumerate(scenes_list):
    # Just get your Open3D point cloud from scenes_list; read into memory for speed reasons
    # o3d.visualization.draw_geometries([geometry])    
    geometry.points = scene.points
    
    if frame == 0:
      vis.add_geometry(geometry)
    else:
      vis.update_geometry(geometry)
  
    vis.poll_events()
    vis.update_renderer()
    
    # Delay
    time.sleep(0.016)

  vis.destroy_window()
  return

def obtain_scenes_details(path2scenes: str) -> list or np.float32 or int:
  
  path2scenes_ext = os.path.join(path2scenes, '*.txt')
  
  # get list of filenames within our scenes list
  # filenames are guaranteed to be of the format "output_FRAME_SENSORRES.txt"
  filenames = [os.path.basename(abs_path) for abs_path in glob.glob(path2scenes_ext)]
  
  res = np.float32(
        float(os.path.splitext(
            (filenames[0].split("_")[-1])   
        )[0])
      )
  
  offset = int(min(filenames, key=lambda x: int((x.split('_'))[1])).split('_')[1])

  mp.freeze_support()
  cores = int((mp.cpu_count()) - 1)

  with mp.Pool(cores) as p:
    result = p.starmap_async(
      open_point_cloud,
      tqdm(
        [
          (path2scenes, frame+offset, res)
          for frame in range(len(filenames))
        ],
        total = len(filenames),
        desc="Reading scenes into memory"
      )
    )
    p.close()
    p.join()

    xyzs = result.get()
    xyzs = [cloud.create_point_cloud() for cloud in xyzs]
    pcds = np.array(xyzs, dtype=object)
    # Vectorize the conversion of our point clouds given in np.ndarray to
    # Open3D point cloud
    
    '''
    # Helper function that will be vectorized
    def convert_cloud_to_o3d(xyz: np.ndarray, pbar) -> o3d.geometry.PointCloud:
      """Helper function to convert a point cloud given in an ndarray
      to a PointCloud object in Open3D. This function will be vectorized for speed

      Args:
          xyz (np.ndarray): Our xyz points of the point cloud.
          
          pbar (tqdm.std.tqdm): Our progress bar for visualization.

      Returns:
          pcd (o3d.geometry.PointCloud): The converted point cloud.
      """
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(xyz)
      pbar.update()
      return pcd

    with tqdm(desc="Preprocessing point clouds", total=xyzs.shape[0], leave=True) as pbar:
      pcds = np.vectorize(convert_cloud_to_o3d)(xyzs, pbar)
      pbar.close()
  ''' 
  return pcds, res, offset

def main():
  args = parse_cmdline_args()
  path_to_scenes = obtain_scene_path(args)
  scenes, res, offset = obtain_scenes_details(path_to_scenes)
  replay_scenes(path_to_scenes, scenes, res, offset)


  # open_point_cloud(path_to_scenes)
  
  return


if __name__ == "__main__":
  main()