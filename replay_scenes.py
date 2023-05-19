import numpy as np
import open3d as o3d
import pandas as pd
import cv2
import tkinter as tk
import sys, os
import glob
import time

from tkinter import Tk
from pathlib import Path
from tqdm import tqdm

from trajectory_tools import parse_cmdline_args, obtain_scene_path

# Global variables for file I/O
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
ROOT2 = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
  sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

class PointCloudOpener:
  # Opens one specified point cloud as a Open3D tensor point cloud for parallelism
  def open_point_cloud(self, path2scenes: str, frame: int, res: np.float32) -> o3d.t.geometry.PointCloud:
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
    # print(scene_name)
    
    # Skip our header, and read only XYZ coordinates
    df = pd.read_csv(path_to_scene, skiprows=0, usecols=[0, 1, 2])
    xyz = df.to_numpy() / 1000
    
    # Create Open3D point cloud object with tensor values.
    # For parallelization, outputs must be able to be serialized 
    pcd = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
    pcd.point.positions = o3d.core.Tensor(xyz, o3d.core.float32, o3d.core.Device("CPU:0"))

    return pcd

# Play our scenes using Open3D's visualizer
def visualize_replay(path2scenes: str, scenes_list: np.ndarray, vehicle_speed: np.float32 = 100, point_density: np.float32 = 1.0) -> list or int:

  # Obtain screen parameters for our video
  root = Tk()
  root.withdraw()
  SCREEN_WIDTH, SCREEN_HEIGHT = root.winfo_screenwidth(), root.winfo_screenheight()

  # Helper function to visualize the replay of our frames in a video format
  def replay_frames():
    
    frames = [] # Store our recorded frames for visualization
    print(f"Visualizing the scenes given by path {path2scenes}")
    
    geometry = o3d.geometry.PointCloud()
    vis.add_geometry(geometry)
    # This is just the coordinate frae for the vectors
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0,0,0])
    vis.add_geometry(coordinate_frame)
    
    # Obtain view control of the visualizer to change POV on setup
    # NOTE Currently, as of 5/19/2023, the get_view_control() method for the open3d.Visualizer class
    # only returns a copy of the view control as opposed to a reference.
    if (o3d.__version__ == "0.17.0"):
      pass
    else:
      ctr = vis.get_view_control()

    for frame, scene in enumerate(tqdm(scenes_list, desc="Replaying scenes")):
      # Just get your Open3D point cloud from scenes_list; read into memory for speed reasons
      # o3d.visualization.draw_geometries([geometry])    
      geometry.points = scene.to_legacy().points  # IF THE SCENE IS IN TENSOR
      # geometry.points = scene.points # Point clouds are preprocessed from tensor to legacy
      
      if frame == 0:
        vis.add_geometry(geometry, reset_bounding_box = True)
      else:
        vis.update_geometry(geometry)
        
      # Set to isometric front view
      if (o3d.__version__ == "0.17.0"): # This probably doesn't work
        # ctr.change_field_of_view(step=50) 
        print(f"WARNING: Setting view control for open3d version {o3d.__version__} does not work! Setting to default.")
        vis.get_view_control().set_front([-1, -1, 1])  
        vis.get_view_control().set_up([0, 0, 1])
        vis.get_view_control().set_lookat([0, 0, 1.8])
        vis.get_view_control().set_zoom(0.3)
      else:    
        # ctr.change_field_of_view(step=50) 
        ctr.set_front([-1, -1, 1])  
        ctr.set_up([0, 0, 1])
        ctr.set_lookat([0, 0, 1.8])
        ctr.set_zoom(0.3)    
      
      vis.poll_events()
      vis.update_renderer()
      
      # Capture the rendered point cloud to an RGB image for video output
      frames.append(np.asarray(vis.capture_screen_float_buffer(do_render=True)))
      
      # Play the scenes as it appears in the vehicle's speed
      time.sleep((1*point_density)/(vehicle_speed/3.6))
    
    return frames, SCREEN_WIDTH, SCREEN_HEIGHT
  
  
  # Example taken from open3d non-blocking visualization...
  vis = o3d.visualization.Visualizer()
  usr_inpt = input(f"Press 'p' to replay {len(scenes_list)} scenes given by {path2scenes} (press 'q' to exit): ")
  if usr_inpt == 'p':
    
    vis.create_window(window_name=f"Scenes of {path2scenes}",
                      width=SCREEN_WIDTH,
                      height=SCREEN_HEIGHT,
                      left=10,
                      top=10,
                      visible=True)
    
    vis.set_full_screen(True)
  
    # View control options (also must be created befoe we can replay our frames)
    # TODO GET VIEW CONTROL OPTIONS TO WORK WITH THE VISUALIZER
    # Render options (must be created before we can replay our frames)
    render_opt = vis.get_render_option() 
    render_opt.point_size = 1.0
    render_opt.show_coordinate_frame = True
    render_opt.background_color = np.array([16/255, 16/255, 16/255]) # 8-bit RGB, (16, 16, 16)
    
    vis.poll_events()
    vis.update_renderer()

    frames = replay_frames()

  elif usr_inpt == 'q':
    return
  else:
    visualize_replay(path2scenes, scenes_list, vehicle_speed, point_density)  

  print("Visualization complete.")
  vis.clear_geometries()
  vis.destroy_window()
  
  return frames

# Create a video with a fixed POV of the replay
def create_video(frames: list, w: int, h: int, path2scenes: str, vehicle_speed: np.float32 = 100, point_density: np.float32 = 1.0) -> None:
  
  
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

  
  # Get filename of the recorded visualization
  filename = f"replay_src={path2scenes[:-1]}.mp4"
  
  output_folder = os.path.join(ROOT2, "visualizations")
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
  output_path = os.path.join(output_folder, filename)
  
  # Configure video writer
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  fps = np.ceil((vehicle_speed/3.6)/(1*point_density))
  writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
  
  # Parameters for annotating text
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 1
  font_thickness = 1
  font_color = (255, 255, 255) # 8-bit RGB
  
  # Now we will create our video
  for frame_i, img in enumerate(tqdm(frames, total=len(frames), desc=f"Writing to video")):
    print(img.shape)
    
    # Open3D normalizes the RGB values from 0 to 1, while OpenCV
    # requires RGB values from 0 to 255 (8-bit RGB)
    img = (img[:,:]*255).astype(np.uint8)
    
    # Annotate our image
    progress_text = f"Frame {str(frame_i+1).rjust(len(str(len(frames))), '0')}/{len(frames)}"
    # Get width and height of thes source text
    progress_text_size = cv2.getTextSize(progress_text, fontFace=font, fontScale=font_scale, thickness=font_thickness)[0]
    progress_xy = (20, progress_text_size[1]+20)
    frame_annotated = annotate_frame(img, progress_text, progress_xy)

    source_text = f"Source: {os.path.basename(os.path.normpath(path2scenes))}"
    # Get width and height of the source text
    source_text_size = cv2.getTextSize(source_text, fontFace=font, fontScale=font_scale, thickness=font_thickness)[0]
    source_xy = (w-source_text_size[0]-20, source_text_size[1]+20)
    frame_annotated = annotate_frame(frame_annotated, source_text, source_xy)
    
    writer.write(cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB))
    
  writer.release()
    
  return

# Read our scenes into memory
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
  
  pcds = Parallel(n_jobs=cores, backend='loky')( # Switched to loky backend to maybe suppress errors?
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

# Main function, everything is called here
def main():
  args = parse_cmdline_args() # From trajectory_tools.py
  path_to_scenes = obtain_scene_path(args)
  scenes = obtain_scenes(path_to_scenes)
  frames, sw, sh = visualize_replay(path_to_scenes, scenes, vehicle_speed=100, point_density=1.0)
  create_video(frames, w=sw, h=sh, path2scenes=path_to_scenes, vehicle_speed=100, point_density=1.0)

  return


if __name__ == "__main__":
  main()