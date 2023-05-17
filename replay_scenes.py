import numpy as np
# import pathos.multiprocessing as mp

import open3d as o3d
import open3d.core as o3c
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

class PointCloudOpener:
  def set_pbar(self, pbar) -> None:
    self.pbar = pbar
  
  def open_point_cloud(self, path2scenes: str, frame: int, res: np.float32) -> o3d.geometry.PointCloud:
    # outputs are in the format output_FRAME_0.11.txt
    # split the first scene file to obtain the sensor resolution...

    scene_name = f"output_{frame}_{res:.2f}.txt"
    path_to_scene = os.path.join(path2scenes, scene_name)
    print(scene_name)
    
    xyzypd = np.genfromtxt(path_to_scene, delimiter=",")
    xyz = np.delete(xyzypd, [3,4,5], axis=1) # We don't want spherical coordinates in our scene data
    xyz = np.delete(xyz, 0, axis=0)          # We don't want our header either since it reads into nan
    xyz /= 1000   # Convert from mm to m
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    self.pbar.update()

    return pcd

'''
def open_point_cloud(path2scenes: str, frame: int, res: np.float32) -> o3d.geometry.PointCloud:
  # outputs are in the format output_FRAME_0.11.txt
  # split the first scene file to obtain the sensor resolution...

  scene_name = f"output_{frame}_{res:.2f}.txt"
  path_to_scene = os.path.join(path2scenes, scene_name)
  
  xyzypd = np.genfromtxt(path_to_scene, delimiter=",")
  xyz = np.delete(xyzypd, [3,4,5], axis=1) # We don't want spherical coordinates in our scene data
  xyz = np.delete(xyz, 0, axis=0)          # We don't want our header either since it reads into nan
  xyz /= 1000   # Convert from mm to m
  
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyz)
  
  pbar.update()

  return pcd
''' 

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

  # Here we will use pathos instead of multiprocessing because of how
  # multiprocessing cannot pickle Open3D point clouds.
  import pathos, multiprocess
  from pathos.helpers import ThreadPool
  from pathos.multiprocessing import ProcessingPool
  import dill

  ''' SMAP, NEED TO PASS POOL INTO IT IN THE FIRST PLACE
  from pathos.maps import Smap
  smap = Smap()
  pcds = smap.__call__(
    open_point_cloud, 
    tqdm(
        [
        (path2scenes, frame+offset, res) for frame in range(len(filenames))
        ],
        total=len(filenames),
        desc="Reading scenes to memory"
      )
    )
  print(pcds)
  '''

  import multiprocessing as mp


  mp.freeze_support()
  cores = min(int((mp.cpu_count()) - 1), len(filenames))
  
  from multiprocess.pool import ThreadPool
  with ThreadPool(processes=cores) as p:

    print(f"Starting parallel pool with {cores} cores...")
    # Arguments of the parallelized for loop
    args = [(path2scenes, frame+offset, res) for frame in range(len(filenames))]
  
    # Map open_point_cloud() for each frame, parallelized using pathos.helpers.
    opener = PointCloudOpener()
    with tqdm(args, total=len(filenames), desc="Reading scenes to memory") as pbar:
      opener.set_pbar(pbar)
      pcds = p.starmap(
        opener.open_point_cloud,
        args
      )
    
    p.close()
    p.join()
    
    print(f"{len(pcds)} scenes were read to memory.")
  

  '''    # CANNOT SERIALIZE
  # with ProcessingPool(nodes=cores) as p:
  with multiprocess.Pool(cores) as p:
    print(f"Starting parallel pool with {cores} cores...")
    
    # Arguments of the parallelized for loop
    args = [(path2scenes, frame+offset, res) for frame in range(len(filenames))]
    #[('sample_scenes/', 245, 0.11), ('sample_scenes/', 246, 0.11), ('sample_scenes/', 247, 0.11), ('sample_scenes/', 248, 0.11), ('sample_scenes/', 249, 0.11), ('sample_scenes/', 250, 0.11), ('sample_scenes/', 251, 0.11), ('sample_scenes/', 252, 0.11), ('sample_scenes/', 253, 0.11), ('sample_scenes/', 254, 0.11), ('sample_scenes/', 255, 0.11)]
    opener = PointCloudOpener()
    # func = lambda x: opener.open_point_cloud(*x)

    with tqdm(args, total=len(filenames), desc="Reading scenes to memory") as pbar:
      # opener.set_pbar(pbar)
      
      pcds = p.starmap(
        opener.open_point_cloud,
        args
      )
      p.close()
      p.join()
      
    print(pcds)
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