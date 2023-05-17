import numpy as np
import open3d as o3d
import pandas as pd
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
  
  def open_point_cloud(self, path2scenes: str, frame: int, res: np.float32) -> o3d.t.geometry.PointCloud:

    scene_name = f"output_{frame}_{res:.2f}.txt"
    path_to_scene = os.path.join(path2scenes, scene_name)
    # print(scene_name)
    
    # Skip our header, and read only XYZ coordinates
    df = pd.read_csv(path_to_scene, skiprows=0, usecols=[0, 1, 2])
    xyz = df.to_numpy() / 1000

    
    # Create Open3D point cloud object with tensor values.
    # For parallelization, outputs must be able to be serialized 
    pcd = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
    pcd.point.positions = o3d.core.Tensor(xyz, o3d.core.float32, o3d.core.Device("CPU:0"))

    return pcd
 
def replay_scenes(path2scenes: str, scenes_list: np.ndarray, vehicle_speed: np.float32, point_density: np.float32) -> None:
  
  usr_inpt = input(f"Press 'p' to replay {len(scenes_list)} scenes given by {path2scenes}: ")
  if usr_inpt == 'p':
    pass
  else:
    replay_scenes(path2scenes, scenes_list, vehicle_speed, point_density)
    
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

  for frame, scene in enumerate(tqdm(scenes_list, desc="Replaying scenes")):
    # Just get your Open3D point cloud from scenes_list; read into memory for speed reasons
    # o3d.visualization.draw_geometries([geometry])    
    geometry.points = scene.to_legacy().points  # IF THE SCENE IS IN TENSOR
    # geometry.points = scene.points # Point clouds are preprocessed from tensor to legacy
    
    if frame == 0:
      vis.add_geometry(geometry)
    else:
      vis.update_geometry(geometry)
  
    vis.poll_events()
    vis.update_renderer()
    # Capture screen of visualizer here (vis.capture_screen_image(PATH2PNG))
    # Get POV of visualizer with 'ctr = vis.get_view_control()?'
    
    # Delay to replay relative to speed
    time.sleep((1*point_density)/(vehicle_speed/3.6))

  print("Visualization complete.")
  vis.destroy_window()
  return

def obtain_scenes(path2scenes: str) -> list:
  
  path2scenes_ext = os.path.join(path2scenes, '*.txt')
  
  # Get list of filenames within our scenes list
  # Filenames are guaranteed to be of the format "output_FRAME_RES.txt"
  filenames = [os.path.basename(abs_path) for abs_path in glob.glob(path2scenes_ext)]
  
  res = np.float32(
        float(os.path.splitext(
            (filenames[0].split("_")[-1])   
        )[0])
      )
  
  # For offsetting frame indexing in case if we are working with padded output
  # Output should usually be padded anyways
  offset = int(min(filenames, key=lambda x: int((x.split('_'))[1])).split('_')[1])

  # Read each of the scenes into memory in parallel
  import joblib
  from joblib import Parallel, delayed
  
  cores = min((joblib.cpu_count() - 1), len(filenames))

  # Create our opener object (for inputs/outputs to be serializable)
  opener = PointCloudOpener()
  # Define the arguments that will be ran upon in parallel.
  args = [(path2scenes, frame+offset, res) for frame in range(len(filenames))]
  
  pcds = Parallel(n_jobs=cores)(
    delayed(opener.open_point_cloud)(arg_path2scenes, arg_frame, arg_res)
    for arg_path2scenes, arg_frame, arg_res in tqdm(args, 
                                                    total=len(filenames), 
                                                    desc=f"Reading scenes to memory in parallel, using {cores} processes")
    )

  print(f"\n{len(pcds)} scenes were read to memory.")
  

  # Now that we have our point clouds in tensor form, we can convert them back into legacy form for replaying.
  # The conversion process should be pretty fast, taking about 0.4ms for a scene with 241350 points.
  # pcds = [pcd.to_legacy() for pcd in tqdm(pcds, desc="Preprocessing scenes", total=len(pcds), leave=True)] 
  
  return pcds

def main():
  args = parse_cmdline_args()
  path_to_scenes = obtain_scene_path(args)
  scenes = obtain_scenes(path_to_scenes)
  replay_scenes(path_to_scenes, scenes, vehicle_speed=100, point_density=1.0)

  return


if __name__ == "__main__":
  main()