import numpy as np
import open3d as o3d
import sys, os
import tempfile
import time

import file_tools
import sensorpoints

from pathlib import Path
from tqdm import tqdm

# NOTE This program will perform everything
# Make it as expandable as possible

# Global variables for file I/O
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
ROOT2 = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
  sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


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

def replay_temp(
  cfg: sensorpoints.SensorConfig, 
  traj: file_tools.Trajectory, 
  road: o3d.geometry.PointCloud,
  src_name: str
  ):

  # Obtain screen parameters for our video
  from tkinter import Tk
  root = Tk()
  root.withdraw()
  SCREEN_WIDTH, SCREEN_HEIGHT = root.winfo_screenwidth(), root.winfo_screenheight()

  # Begin our visualization...
  ## Setup our visualizer
  vis = o3d.visualization.Visualizer()
  
  vis.create_window(window_name=f"Replay of sensor FOVs on {src_name}",
                    width=SCREEN_WIDTH,
                    height=SCREEN_HEIGHT
                    )
  
  vis.set_full_screen(True)
  # Obtain view control of the visualizer to change POV on setup
  # NOTE Currently, as of 5/19/2023, the get_view_control() method for the open3d.Visualizer class
  # only returns a copy of the view control as opposed to a reference.
  if (o3d.__version__ == "0.17.0"):
    pass
  else:
    ctr = vis.get_view_control()
  
  render_opt = vis.get_render_option()
  render_opt.point_size = 1.0
  render_opt.show_coordinate_frame = True
  render_opt.background_color = np.array([16/255, 16/255, 16/255]) # 8-bit RGB, (16, 16, 16)
  
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
    geometry.points = o3d.utility.Vector3dVector(aligned_fov_points[0]) # [0] temp for single_sensor
    geometry.colors = o3d.utility.Vector3dVector(np.ones((aligned_fov_points[0].shape[0],3),dtype=np.float64)) # [0] temp for single_sensor
    
    centerpt = traj.getRoadPoints()[np.floor(num_points/2).astype(int), :].T.reshape((3,1))
    
    # Set the view
    ctr.set_front([0, 1, 1])  
    ctr.set_up([0, 0, 1])
    ctr.set_lookat(centerpt)
    ctr.set_zoom(0.425) 
    
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

def main():
  args = file_tools.parse_cmdline_args()
  traj = file_tools.obtain_trajectory_details(args)
  cfg  = sensorpoints.open_sensor_config_file(args)
  road = file_tools.open_las(args)
  
  road_o3d, src_name = las2o3d_pcd(road)
  
  # Replays the sensor FOV on the road
  replay_temp(cfg, traj, road_o3d, src_name)
  
  # Replays the Vista outputs
  
  return

if __name__ == "__main__":
  main()