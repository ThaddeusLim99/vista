import numpy as np
import open3d as o3d
import pandas as pd
import sys, os
import tempfile
import time
import cv2

import file_tools
import sensorpoints

from pathlib import Path
from tqdm import tqdm

# NOTE This program will perform all of the visualizations
# (driver POV, real-time visualization of FOV and graphs) and write to video
# Make it as expandable as possible

# Global variables for file I/O
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
ROOT2 = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
  sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

class VistaSceneOpener:
  # Opens one specified point cloud as a Open3D tensor point cloud for parallelism
  def open_scene(self, path2scenes: str, frame: int, res: np.float32) -> o3d.t.geometry.PointCloud:
    """Reads a specified Vista scene from a path into memory.
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
    # print(scene_name)
    
    # Skip our header, and read only XYZ coordinates
    df = pd.read_csv(path_to_scene, skiprows=0, usecols=[0, 1, 2])
    xyz = df.to_numpy() / 1000
    
    # Create Open3D point cloud object with tensor values.
    # For parallelization, outputs must be able to be serialized because Python sucks. 
    pcd = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
    pcd.point.positions = o3d.core.Tensor(xyz, o3d.core.float32, o3d.core.Device("CPU:0"))

    return pcd

def las2o3d_pcd(las: file_tools.LasPointCloud) -> o3d.geometry.PointCloud:
  """Reads the road section into memory such that we can visualize
  the sensor FOV as it goes through the road section.

  Args:
      las (file_tools.LasPointCloud): Our .las point cloud to read values from

  Returns:
      pcd (o3d.geometry.PointCloud): Our point cloud, replay compatible.
      
      filename (str): The respective filename of the las file.
  """
  
  pcd = o3d.geometry.PointCloud()
  
  las_xyz = np.vstack((las.getX(), las.getY(), las.getZ())).T
  pcd.points = o3d.utility.Vector3dVector(las_xyz)
  print("Points loaded.")
  
  import matplotlib
  from matplotlib import colors
  
  las_intensity = las.getIntensity()
  
  # Normalize intensity values to [0, 1]
  las_intensity = (las_intensity - np.min(las_intensity)) / (np.max(las_intensity) - np.min(las_intensity))
  # Assign RGB values to our normalized values
  cmap = matplotlib.cm.gray
  las_rgb = cmap(las_intensity)[:,:-1] # cmap(las_intensity) returns RGBA, cut alpha channel
  pcd.colors = o3d.utility.Vector3dVector(las_rgb)
  print("Intensity colors loaded.")
  
  filename = las.getLasFileName()
  
  return pcd, filename

# Replays the sensor FOV as our simulated vehicle goes down the road section itself.
def replay_sensor_fov(
  cfg: sensorpoints.SensorConfig, 
  traj: file_tools.Trajectory, 
  road: o3d.geometry.PointCloud,
  src_name: str,
  temp_dir: tempfile.TemporaryDirectory
  ):

  # Obtain screen parameters for our video
  from tkinter import Tk
  root = Tk()
  root.withdraw()
  SCREEN_WIDTH, SCREEN_HEIGHT = root.winfo_screenwidth(), root.winfo_screenheight()

  # Setup our visualizer
  vis = o3d.visualization.Visualizer()
  
  vis.create_window(window_name=f"Replay of sensor FOVs on {src_name}",
                    width=SCREEN_WIDTH,
                    height=SCREEN_HEIGHT
                    )
  
  vis.set_full_screen(True) # Full screen to capture full view
  
  # Obtain view control of the visualizer to change POV on setup
  # NOTE Currently, as of 5/19/2023, the get_view_control() method for the open3d.Visualizer class
  # only returns a copy of the view control as opposed to a reference.
  if (o3d.__version__ == "0.17.0"):
    pass
  else:
    ctr = vis.get_view_control()
  
  # Configure our render option
  render_opt = vis.get_render_option()
  render_opt.point_size = 1.0
  render_opt.show_coordinate_frame = True # Does this even work
  render_opt.background_color = np.array([16/255, 16/255, 16/255]) # 8-bit RGB, (16, 16, 16)
  
  # Initalize geometries
  vis.add_geometry(road)
  
  geometry = o3d.geometry.PointCloud()
  vis.add_geometry(geometry)

  vis.poll_events()
  vis.update_renderer()

  # Begin our replay of the sensor FOV
  num_points = traj.getNumPoints()

  for frame in range(num_points):

    # Get sensor FOV
    fov_points = sensorpoints.generate_sensor_points(cfg)
    aligned_fov_points, _ = sensorpoints.align_sensor_points(fov_points, traj, frame)
    
    # TODO Accomodate multi-sensor configurations
    # (first do this through sensorpoints.py)
    geometry.points = o3d.utility.Vector3dVector(aligned_fov_points[0]) # [0] temp for single_sensor
    geometry.colors = o3d.utility.Vector3dVector(np.ones((aligned_fov_points[0].shape[0],3),dtype=np.float64)) # [0] temp for single_sensor
    
    centerpt = traj.getRoadPoints()[np.floor(num_points/2).astype(int), :].T.reshape((3,1))
    
    # Set the view 
    ctr.set_front([0, 1, 1])  
    ctr.set_up([0, 0, 1])
    ctr.set_lookat(centerpt)
    ctr.set_zoom(0.40) 
    
    # Then update the visualizer
    if frame == 0:
      vis.add_geometry(geometry, reset_bounding_box=True)
    else:
      vis.update_geometry(geometry)
    
    vis.poll_events()
    vis.update_renderer()
    
    # Save rendered scene to an image so that we can write it to a video
    #img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    #img = (img[:,:]*255).astype(np.uint8) # Normalize RGB to 8-bit
    
    
  # Done visualizing the scenes
  vis.clear_geometries()
  vis.destroy_window()
  
  
  
  return

# TODO Replays the Vista outputs as our simulated vehicle goes down the road section itself.
def replay_vista():
  return

# TODO Replays the data rate graphs being drawn as our simulated vehicle goes down the road section itself.
def replay_graphical():
  # Shouldn't be too bad
  # set axis range
  # draw real time and then save corresponding frames to tmp dir
  return

# TODO Stitches the saved frames together into one video.
def stitch_frames():
  return

# Read our scenes into memory
def obtain_scenes(path2scenes: str) -> list:
  
  import glob
  
  path2scenes_ext = os.path.join(path2scenes, '*.txt')
  
  # Get list of filenames within our scenes list
  # Filenames are guaranteed to be of the format "output_FRAME_RES.txt"
  filenames = [os.path.basename(abs_path) for abs_path in glob.glob(path2scenes_ext)]
  
  # Obtain sensor resolution
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
  opener = VistaSceneOpener()
  
  # Define the arguments that will be ran upon in parallel
  args = [(path2scenes, frame+offset, res) for frame in range(len(filenames))]
  
  pcds = Parallel(n_jobs=cores, backend='loky')( # Switched to loky backend to maybe suppress errors?
    delayed(opener.open_scene)(arg_path2scenes, arg_frame, arg_res)
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
  # Prepare relevant files for visualization
  args = file_tools.parse_cmdline_args()
  
  traj = file_tools.obtain_trajectory_details(args)
  cfg  = sensorpoints.open_sensor_config_file(args)
  road = file_tools.open_las(args)
  path2scenes = file_tools.open_scene_path(args)
  
  # Create a directory where we temporarily store all of our frames 
  # for writing to video
  temp_dir = tempfile.TemporaryDirectory(dir=ROOT2)
  #subdirectories for organization
  #os.makedirs(os.path.join(tempdir.name, "fov"))
  #os.makedirs(os.path.join(tempdir.name, "vista"))
  #os.makedirs(os.path.join(tempdir.name, "graphical"))
  
  # Replays the sensor FOV on the road
  road_o3d, src_name = las2o3d_pcd(road)
  replay_sensor_fov(cfg, traj, road_o3d, src_name, temp_dir)
  
  # Replays the Vista outputs
  scenes = obtain_scenes(path2scenes)
  
  # Writes our frames into a video
  #stitch_frames()
  temp_dir.cleanup()
  
  return

if __name__ == "__main__":
  main()