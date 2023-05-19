import pandas as pd
import numpy as np
import torch
# import multiprocessing as mp
import os
import sys

from trajectory_tools import obtain_scene_path, parse_cmdline_args
from tqdm import tqdm
from pathlib import Path


# Global variables for file I/O
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
ROOT2 = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
  sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def trim_scene(path2scenes: str, path2outputs: str, frame: int, res: np.float32, ytol: tuple) -> int:
  """Reads a specified point cloud from a path into memory.
  This is called in the parallelized loop in obtain_scenes().

  Args:
      path2scenes (str): The path to the folder containing the scenes.
      frame (int): The frame of the particular scene.
      res (np.float32): The resolution of the sensor at which the scene was recorded.
      This should be given in the filename, where scene names are guaranteed to be
      "output_<FRAME>_<RES>.txt".

  Returns:
      pcd (o3d.t.geometry.PointCloud): Our point cloud, in tensor format.
  """



  scene_name = f"output_{frame}_{res:.2f}.txt"
  path_to_scene = os.path.join(path2scenes, scene_name)
  
  # Skip our header, and read only XYZ coordinates
  # xyzypd = pd.read_csv(path_to_scene, skiprows=0, usecols=[0, 1, 2]).to_numpy()
  # xyzypd = np.delete(xyzypd, 0, axis=0)
  xyzypd = pd.read_csv(path_to_scene).to_numpy()
  # print(xyzypd.shape)
  
  xyzypd  = xyzypd[(xyzypd[:,1] > ytol[0]*1000), :]
  trimmed = xyzypd[(xyzypd[:,1] < ytol[1]*1000), :]
  # print(trimmed.shape)
  
  df = pd.DataFrame(trimmed)
  df.columns = ["x", "y", "z", "yaw", "pitch", "depth"]
  df.to_csv(os.path.join(path2outputs, scene_name), index=False)

  return 0
  
def reduce_scenes(path2scenes: str):

  import joblib
  import glob
  from joblib import Parallel, delayed

  path2scenes_ext = os.path.join(path2scenes, '*.txt')
  filenames = [os.path.basename(abs_path) for abs_path in glob.glob(path2scenes_ext)]

  res = np.float32(
        float(os.path.splitext(
            (filenames[0].split("_")[-1])   
        )[0])
      )

  # For offsetting frame indexing in case if we are working with padded output
  # Output should usually be padded anyways
  offset = int(min(filenames, key=lambda x: int((x.split('_'))[1])).split('_')[1])
  
  ytol = (-8, 8)
  
  # Define our output directory
  output_folder = os.path.join(ROOT2, "trimmed")
  if not os.path.exists(output_folder):
    os.makedirs(output_folder) 
  
  # Define our road-section specific output folder
  output_subfolder = os.path.join(output_folder, f"{os.path.basename(os.path.normpath(path2scenes))}_y={ytol[0]}to{ytol[1]}m")
  if not os.path.exists(output_subfolder):
    os.makedirs(output_subfolder) 
  
  # Define function arguments that we will iterate through
  args = [(path2scenes, output_subfolder, frame+offset, res, ytol) for frame in range(len(filenames))]


  

  cores = min((joblib.cpu_count() - 1), len(filenames))
  
  _ = Parallel(n_jobs=cores, backend='loky')( # Switched to loky backend to maybe suppress errors?
    delayed(trim_scene)(arg_path2scenes, arg_path2outputs, arg_frame, arg_res, arg_ytol)
    for arg_path2scenes, arg_path2outputs, arg_frame, arg_res, arg_ytol in tqdm(args, 
                                                    total=len(filenames), 
                                                    desc=f"Trimming scenes, using {cores} processes")
    )

  '''
  import multiprocessing as mp
  
  cores = mp.cpu_count() - 1
  with mp.Pool(cores) as p:
    _ = p.starmap(trim_scene,
                  tqdm(args,
                       total=len(filenames), 
                       desc=f"Trimming scenes, using {cores} processes")
                  )
    p.close()
    p.join()
  '''
  return


def main():
  args = parse_cmdline_args()
  path2scenes = obtain_scene_path(args)
  reduce_scenes(path2scenes)
  
  
  '''
  # Temp debug...
  trimmedtemp = trim_scene(path2scenes, 245, 0.11, (-8, 8))
  
  import open3d as o3d
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(trimmedtemp)
  o3d.visualization.draw_geometries([pcd])
  '''
  
  return

if __name__ == "__main__":
  main()
