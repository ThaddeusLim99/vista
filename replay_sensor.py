import numpy as np
import open3d as o3d
import pandas as pd
import sys, os
import tempfile
import glob
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

##### Helper functions below #####

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
  print("\nPoints loaded.")
  
  import matplotlib
  
  las_intensity = las.getIntensity()
  
  # Normalize intensity values to [0, 1], then assign RGB values
  normalizer = matplotlib.colors.Normalize(np.min(las_intensity), np.max(las_intensity))
  las_rgb = matplotlib.cm.gray(normalizer(las_intensity))[:,:-1]
  pcd.colors = o3d.utility.Vector3dVector(las_rgb) # cmap(las_intensity) returns RGBA, cut alpha channel
  print("Intensity colors loaded.")
  
  filename = las.getLasFileName()
  
  return pcd, filename

#FIXME Sometimes the animation does not play at all
# Replays the sensor FOV as our simulated vehicle goes down the road section itself.
def render_sensor_fov(
  cfg: sensorpoints.SensorConfig, 
  traj: file_tools.Trajectory, 
  road: o3d.geometry.PointCloud,
  src_name: str,
  temp_dir: tempfile.TemporaryDirectory,
  screen_width: int,
  screen_height: int,
  offset: int
  ) -> None:

  # Helper function to set the visualizer POV
  def set_visualizer_pov(mode: str) -> None:
    if mode == 'center':
      centerpt = traj.getRoadPoints()[np.floor(num_points/2).astype(int), :].T.reshape((3,1))
      # Set the view at the center, top down
      ctr.set_front([0, 1, 1])
      ctr.set_up([0, 0, 1])
      ctr.set_lookat(centerpt)
      ctr.set_zoom(0.3125)
    elif mode == "follow":
      # Follow the vehicle as it goes through the map
      ctr.set_front([0, 1, 1])  
      ctr.set_up([0, 0, 1])
      ctr.set_lookat(traj.getRoadPoints()[frame, :]) # Center the view around the sensor FOV
      ctr.set_zoom(0.3125)
 
  # Setup our visualizer
  vis = o3d.visualization.Visualizer()
  # NOTE open3d.visualization.rendering.OffscreenRenderer can probably be used here
  # instead of calling a GUI visualizer
  
  vis.create_window(window_name=f"Replay of sensor FOVs on {src_name}",
                    width=screen_width,
                    height=screen_height
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
  fov_points = sensorpoints.generate_sensor_points(cfg)
  subfolder = os.path.join(temp_dir.name, 'fov')
  if not os.path.exists(subfolder):
    os.makedirs(subfolder)
  
  for frame in range(0+offset, num_points-offset):

    # Get sensor FOV
    aligned_fov_points, _ = sensorpoints.align_sensor_points(fov_points, traj, frame)
    
    # TODO Accomodate multi-sensor configurations
    # (first do this through sensorpoints.py, note that fov_points is a list of all the XYZ sensor points)
    geometry.points = o3d.utility.Vector3dVector(aligned_fov_points[0]) # [0] temp for single_sensor
    geometry.colors = o3d.utility.Vector3dVector(np.ones((aligned_fov_points[0].shape[0],3),dtype=np.float64)) # [0] temp for single_sensor
    
    # Set the view at the center
    set_visualizer_pov('center')
    
    # Then update the visualizer
    if frame == 0+offset:
      vis.add_geometry(geometry, reset_bounding_box=False)
    else:
      vis.update_geometry(geometry)
    
    vis.poll_events()
    vis.update_renderer()
    
    # Save rendered scene to an image so that we can write it to a video
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    img = (img[:,:]*255).astype(np.uint8) # Normalize RGB to 8-bit
    
    cv2.imwrite(filename=os.path.join(subfolder, f"{frame-offset}.png"), img=img)
    
  print(f"\nFOV rendering on road complete.")
  #vis.clear_geometries()
  vis.destroy_window()

  return

#FIXME Black screen with visualizer. replay_scenes has a working implementation
# Replays the Vista outputs as our simulated vehicle goes down the road section itself.
def render_vista(
  path2scenes: str, 
  scenes_list: list,
  temp_dir: tempfile.TemporaryDirectory,
  screen_width: int,
  screen_height: int
  ) -> None:
  
  # Helper function to set the visualizer POV
  def set_visualizer_pov(mode: str) -> None:
    if mode == 'pov':
      # Set the view to be from the vehicle's POV
      # TODO This should be properly centered, the values here are eyeballed
      ctr.set_front([-1, 0, 0])  
      ctr.set_up([0, 0, 1])
      ctr.set_lookat([18.5, 0, 1.8])
      ctr.set_zoom(0.025)
    elif mode == "isometric":
      # Set the view to be isometric front
      ctr.set_front([-1, -1, 1])  
      ctr.set_up([0, 0, 1])
      ctr.set_lookat([0, 0, 1.8])
      ctr.set_zoom(0.3)           
  
  # Setup our visualizer
  vis = o3d.visualization.Visualizer()
  # NOTE open3d.visualization.rendering.OffscreenRenderer can probably be used here
  # instead of calling a GUI visualizer
  vis.create_window(window_name=f"Replay of Vista scenes",
                    width=screen_width,
                    height=screen_height,
                    left=10,
                    top=10,
                    visible=True)
  
  vis.set_full_screen(True) # Full screen to capture full view

  # Configure our render option
  render_opt = vis.get_render_option()
  render_opt.point_size = 1.0
  render_opt.show_coordinate_frame = True # Does this even work
  render_opt.background_color = np.array([16/255, 16/255, 16/255]) # 8-bit RGB, (16, 16, 16)
  
  vis.poll_events()
  vis.update_renderer()

  geometry = o3d.geometry.PointCloud()
  vis.add_geometry(geometry)
  
  # Obtain view control of the visualizer to change POV on setup
  # NOTE Currently, as of 5/19/2023, the get_view_control() method for the open3d.Visualizer class
  # only returns a copy of the view control as opposed to a reference.
  if (o3d.__version__ == "0.17.0"):
    pass
  else:
    ctr = vis.get_view_control()

  # Begin our replay of the sensor FOV
  subfolder = os.path.join(temp_dir.name, 'vista')
  if not os.path.exists(subfolder):
    os.makedirs(subfolder)
    
  for frame, scene in enumerate(scenes_list):
    # Our loaded scenes are given in o3d.t.geometry.PointCloud,
    # convert back to legacy to visualize
    geometry.points = scene.to_legacy().points
    
    if frame == 0:
      vis.add_geometry(geometry, reset_bounding_box = False)
    else:
      vis.update_geometry(geometry)
      
    # Set the view from the vehicle's POV
    set_visualizer_pov('pov')

  
    # Update the renderer
    vis.poll_events()
    vis.update_renderer()
    
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    img = (img[:,:]*255).astype(np.uint8) # Normalize RGB to 8-bit
    
    cv2.imwrite(filename=os.path.join(subfolder, f"{frame}.png"), img=img)

  print(f"Scene rendering complete.")
  vis.clear_geometries()
  vis.destroy_window()
  
  return

# TODO Replays the data rate graphs being drawn as our simulated vehicle goes down the road section itself.
def render_graphical():
  # Shouldn't be too bad
  # set axis range
  # draw real time and then save corresponding frames to tmp dir
  # given the vista outputs draw graphs for each of the vista outputs
  #os.makedirs(os.path.join(tempdir.name, "graphical"))
  return

# TODO Stitches the saved frames together into one video.
def stitch_frames(temp_dir: tempfile.TemporaryDirectory):
  
  # Stitch, frames can either be from any of the fov, vista, or graphical outputs
  frames = {}
  
  # Read subdirectories from our temporary directory
  subfolder_paths = glob.glob(f'{temp_dir.name}/*/')
  subfolder_paths = [os.path.normpath(path) for path in subfolder_paths]
  # subfolder_names = [os.path.basename(path) for path in subfolder_paths]

  # Get paths to all of the respective images for each of the captures 
  for subfolder_path in subfolder_paths:
    subfolder_name = os.path.basename(subfolder_path) # 'fov' or 'graphical' or 'vista'

    # Obtain paths to our saved frames
    path2temp_ext = os.path.join(subfolder_path, '*.png')
    filenames = [os.path.basename(abs_path) for abs_path in glob.glob(path2temp_ext)]
    filenames = sorted(filenames, key=lambda f: int(os.path.splitext(f)[0]))
    
    frames[subfolder_name] = filenames
    
  # Now that we have our frames from each respective capture, we can now stitch the video together
  # The question is how would I stitch each frame together...
  # Resize, place, annotate, etc


  # Helper function to annotate text onto a frame
  def annotate_frame(image: np.ndarray, text: str, coord: tuple) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    annotated_image = cv2.putText(
      img=image, 
      text=text, 
      org=coord, 
      fontFace=font,            # Defined below
      fontScale=font_scale,     #
      color=font_color,         #
      thickness=font_thickness, #
      lineType=cv2.LINE_AA
      )
    
    return annotated_image
  
  return

# Read our scenes into memory
def obtain_scenes(path2scenes: str) -> list:
  
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

# Obtains screen size (width, height) in pixels
def obtain_screen_size() -> tuple:
  # Obtain screen parameters for our video
  from tkinter import Tk
  root = Tk()
  root.withdraw()
  
  SCREEN_WIDTH, SCREEN_HEIGHT = root.winfo_screenwidth(), root.winfo_screenheight()
  
  return (SCREEN_WIDTH, SCREEN_HEIGHT)

def check_for_padded(path2scenes: str) -> int:
  path2scenes_ext = os.path.join(path2scenes, '*.txt')
  filenames = [os.path.basename(abs_path) for abs_path in glob.glob(path2scenes_ext)]
  offset = int(min(filenames, key=lambda x: int((x.split('_'))[1])).split('_')[1])
  
  return offset

##### Driver functions below #####

def run_sensor_fov(
  road: file_tools.LasPointCloud,
  cfg: sensorpoints.SensorConfig,
  traj: file_tools.Trajectory,
  temp_dir: tempfile.TemporaryDirectory,
  screen_wh: tuple,
  offset: int
  ) -> None:

  road_o3d, src_name = las2o3d_pcd(road)
  render_sensor_fov(cfg=cfg, 
                    traj=traj, 
                    road=road_o3d, 
                    src_name=src_name, 
                    temp_dir=temp_dir,
                    screen_width=screen_wh[0],
                    screen_height=screen_wh[1],
                    offset=offset
                    )
  return

def run_vista(
  path2scenes: str, 
  temp_dir: tempfile.TemporaryDirectory, 
  screen_wh: tuple) -> None:

  scenes = obtain_scenes(path2scenes)
  render_vista(path2scenes=path2scenes,
               scenes_list=scenes,
               temp_dir=temp_dir,
               screen_width=screen_wh[0],
               screen_height=screen_wh[1]
               )
  return


def main():
  # Prepare relevant files for visualization
  args = file_tools.parse_cmdline_args()
  
  traj = file_tools.obtain_trajectory_details(args)
  cfg  = sensorpoints.open_sensor_config_file(args)
  road = file_tools.open_las(args)
  path2scenes = file_tools.obtain_scene_path(args)
  
  # Create a temporary directory to store captured frames
  temp_dir = tempfile.TemporaryDirectory(dir=ROOT2)
  screen_wh = obtain_screen_size()
  frame_offset = check_for_padded(path2scenes)
  
  # Runs and captures the sensor FOV on the road
  # run_sensor_fov(road=road, cfg=cfg, traj=traj, temp_dir=temp_dir, screen_wh=screen_wh, offset=offset)
  
  # Runs and captures the Vista scenes
  run_vista(path2scenes, temp_dir, screen_wh)
  #import replay_scenes
  #slist = obtain_scenes(path2scenes)
  #_,_,_ = replay_scenes.visualize_replay(path2scenes=path2scenes, scenes_list=slist, vehicle_speed=100, point_density=1)
  
  # Writes our frames into a video
  #stitch_frames()
  
  # Remove temporary directory for our files
  temp_dir.cleanup()
  
  return

if __name__ == "__main__":
  main()