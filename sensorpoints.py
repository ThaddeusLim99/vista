import numpy as np
import argparse
import json
import sys
import os
import laspy
import tkinter as tk
from tkinter import Tk
from pathlib import Path
from time import perf_counter

import gen_traj

'''
Sensor points
Eric Cheng
2023-05-04
Generate sensor points and write to a las file for visualization
at a specific road point, given a trajectory.
Based off of the code by JM, and ZP, but with Python for OS compatibility.
'''

# Global variables for file I/O
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
ROOT2 = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
  sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

class SensorConfig:
  '''
  Container class for the sensor configuration.
  '''
  def __init__(self, numberSensors, horizAngRes, verticAngRes, e_low, e_high, a_low, a_high, r_low, r_high):
    self.numberSensors = numberSensors;
    self.horizAngRes = horizAngRes;
    self.verticAngRes = verticAngRes;
    self.e_low = e_low;
    self.e_high = e_high;
    self.a_low = a_low;
    self.a_high = a_high;
    self.r_low = r_low;
    self.r_high = r_high;
  pass

  sensor_config_filename = None

  # We shouldn't need setters, let alone getters since we are 
  # creating only one container object, but I did it just in case.
  def getNumberSensors(self):
    return self.numberSensors
  def getHorizAngRes(self):
    return self.horizAngRes
  def getVerticAngRes(self):
    return self.verticAngRes
  def getELow(self):
    return self.e_low
  def getEHigh(self):
    return self.e_high
  def getALow(self):
    return self.a_low
  def getAHigh(self):
    return self.a_high
  def getRLow(self):
    return self.r_low
  def getRHigh(self):
    return self.r_high

#TODO Obtain the trajectory using trajectory_tools.py instead of locally
# defining the functions
class Trajectory:
  """Container class for the trajectory. 
  """
  def __init__(self, observer_points: np.ndarray, forwards: np.ndarray, upwards: np.ndarray, leftwards: np.ndarray) -> None:
    self.__observer_points = observer_points
    self.__forwards = forwards
    self.__upwards = upwards
    self.__leftwards = leftwards
    pass
  
  def getObserverPoints(self) -> np.ndarray:
    return self.__observer_points
  def getForwards(self) -> np.ndarray:
    return self.__forwards
  def getUpwards(self) -> np.ndarray:
    return self.__upwards
  def getLeftwards(self) -> np.ndarray:
    return self.__leftwards

def open_sensor_config_file(args: argparse.Namespace) -> SensorConfig:
  """Opens the sensor configuration file from command-line argument or
  through UI.

  Args:
      args (argparse.Namespace): Contains the command-line arguments.

  Returns:
      cfg (SensorConfig): Container class containing the sensor configuration
      parameters.
  """
  print("Opening sensor file!")
  # read the sensor config file and save the params
  if args.config == None:
    # Manually get sensor configuration file
    Tk().withdraw();
    sensorcon_filepath = tk.filedialog.askopenfilename(
      filetypes = [
        (".json files", "*.json"),
        ("All files", "*")
        ],
      initialdir = os.path.join(ROOT2, 'sensors/'),
      title = "Please select the sensor configuration file"

    )
    print(f"You have chosen to open the sensor file:\n{sensorcon_filepath}")
    
  else:
    sensorcon_filepath = args.config
    print(f"Using predefined sensor file: {os.path.basename(sensorcon_filepath)}")
  
  tStart = perf_counter()
  
  with open(sensorcon_filepath, 'r') as f:
    data = f.read()

  sensor_cfg_dict = json.loads(data);

  # Create container object
  cfg = SensorConfig(
    sensor_cfg_dict["numberSensors"],
    sensor_cfg_dict["horizAngRes"],
    sensor_cfg_dict["verticAngRes"],
    sensor_cfg_dict["e_low"],
    sensor_cfg_dict["e_high"],
    sensor_cfg_dict["a_low"],
    sensor_cfg_dict["a_high"],
    sensor_cfg_dict["r_low"],
    sensor_cfg_dict["r_high"]
  )
  
  cfg.sensor_config_filename = os.path.basename(sensorcon_filepath)

  tStop = perf_counter()

  print(f"Loading took {(tStop-tStart):.2f}s.")

  return cfg

# Converted from generate_sensor_points_cell.m
#TODO Expand the sensor point generation for multiple sensors...
def generate_sensor_points(sensor_config: SensorConfig) -> list:
  """Creates a set of XYZ points that represent the FOV of the sensor
  configuration in meters, for each sensor.

  Args:
      sensor_config (SensorConfig): Container class for
      the sensor configuration.

  Returns:
      points (list): List containing the XYZ points of the form 
      np.ndarray that make up the FOV of each sensor.
  """

  print(f"\nGenerating FOV points for {sensor_config.getNumberSensors()} sensor(s)!")
  tStart = perf_counter()

  # Override the resolution
  verticalRes = 2
  horizontalRes = 2

  # Gammas correspond to vertical angles
  total_gammas = np.int32(
    np.floor(
      np.abs(sensor_config.getEHigh()-sensor_config.getELow())/verticalRes
      )
    )
  gammas = np.linspace(
    sensor_config.getEHigh(),
    sensor_config.getELow(),
    total_gammas,
  ).reshape(total_gammas, 1)

  ## Now create the points that make up the surfaces of the sensor FOV
  points = [] # Container of points for each sensor
  # For now we will only care about one sensor
  # TODO Expand code for multi-sensor configurations (tesla_day.json)
  for i in range(sensor_config.getNumberSensors()):
    
    # For multisensor configurations in json files, each field is a list.
    # This shouldn't be too bad to expand for other sensors
    
    # Thetas correspond to horizontal angles
    total_thetas = np.int32(
      np.floor(
        np.abs(sensor_config.getAHigh()-sensor_config.getALow())/horizontalRes
        )
      )
    thetas = np.linspace(
      -sensor_config.getALow(),
      -sensor_config.getAHigh(),
      total_thetas,
    ).reshape(1, total_thetas)
    
    # Obtain XYZ points that will make up the front of the FOV
    fronts = np.hstack( # X Y Z
        (
          np.reshape(np.cos(np.deg2rad(gammas))*np.cos(np.deg2rad(thetas))*sensor_config.getRHigh(), (total_gammas*total_thetas, 1), order='F'),
          np.reshape(np.cos(np.deg2rad(gammas))*np.sin(np.deg2rad(thetas))*sensor_config.getRHigh(), (total_gammas*total_thetas, 1), order='F'),
          np.reshape(np.sin(np.deg2rad(gammas))*np.ones(thetas.shape[1])  *sensor_config.getRHigh(), (total_gammas*total_thetas, 1), order='F')
        )
      )
    
    ## Obtain ranges
    total_vert_ranges = np.int32(
      np.floor(
        1/np.sin(np.deg2rad(verticalRes))
        )
      )
    
    vert_ranges = np.linspace(
      0,
      sensor_config.getRHigh(), 
      total_vert_ranges
      ).reshape(total_vert_ranges, 1)
    
    total_horz_ranges = np.int32(
      np.floor(
        1/np.sin(np.deg2rad(horizontalRes))
        )
      )
    
    horz_ranges = np.linspace(
      0,
      sensor_config.getRHigh(),
      total_horz_ranges
    ).reshape(1, total_horz_ranges)
    
    # i'm not even sure if this this right but we will have to see
    if ((sensor_config.getALow() != -180) or (sensor_config.getAHigh() != 180)):
      
      # Obtain XYZ points that will make up the left side of the FOV
      left_side = np.hstack( # X Y Z
        (
          np.reshape(np.cos(np.deg2rad(gammas))*horz_ranges*np.cos(np.deg2rad(-sensor_config.getALow())), (total_gammas*total_horz_ranges, 1), order='F'),
          np.reshape(np.cos(np.deg2rad(gammas))*horz_ranges*np.sin(np.deg2rad(-sensor_config.getALow())), (total_gammas*total_horz_ranges, 1), order='F'),
          np.reshape(np.sin(np.deg2rad(gammas))*horz_ranges, (total_gammas*total_horz_ranges, 1), order='F')
          )
        )
      
      # Obtain XYZ points that will make up the right side of the FOV
      right_side = np.hstack( # X Y Z
         (
           np.reshape(np.cos(np.deg2rad(gammas))*horz_ranges*np.cos(np.deg2rad(-sensor_config.getAHigh())), (total_gammas*total_horz_ranges, 1), order='F'),
           np.reshape(np.cos(np.deg2rad(gammas))*horz_ranges*np.sin(np.deg2rad(-sensor_config.getAHigh())), (total_gammas*total_horz_ranges, 1), order='F'),
           np.reshape(np.sin(np.deg2rad(gammas))*horz_ranges,(total_gammas*total_horz_ranges, 1), order='F')
           )
         )       
      
    else:
      # Sensor FOV already covers from -180 to 180 degrees, we don't need left and right sides
      left_side = np.zeros((0, 3))
      right_side = left_side
    
    # i'm not even sure if this this right but we will have to see
    # Obtain XYZ points that will make up the top side of the FOV
    top_side = np.hstack( # X Y Z
      (
        np.reshape(np.cos(np.deg2rad(sensor_config.getEHigh()))*vert_ranges*np.cos(np.deg2rad(thetas)), (total_thetas*total_vert_ranges, 1), order='F'),
        np.reshape(np.cos(np.deg2rad(sensor_config.getEHigh()))*vert_ranges*np.sin(np.deg2rad(thetas)), (total_thetas*total_vert_ranges, 1), order='F'),
        np.reshape(np.sin(np.deg2rad(sensor_config.getEHigh()))*vert_ranges*np.ones(thetas.shape[1]),   (total_thetas*total_vert_ranges, 1), order='F')
        )
      )

    # i'm not even sure if this this right but we will have to see
    # Obtain XYZ points that will make up the bottom side of the FOV
    bot_side = np.hstack( # X Y Z
      (
        np.reshape(np.cos(np.deg2rad(sensor_config.getELow()))*vert_ranges*np.cos(np.deg2rad(thetas)), (total_thetas*total_vert_ranges, 1), order='F'),
        np.reshape(np.cos(np.deg2rad(sensor_config.getELow()))*vert_ranges*np.sin(np.deg2rad(thetas)), (total_thetas*total_vert_ranges, 1), order='F'),
        np.reshape(np.sin(np.deg2rad(sensor_config.getELow()))*vert_ranges*np.ones(thetas.shape[1]),   (total_thetas*total_vert_ranges, 1), order='F')
        )
      )
    
    # Now that we have all of the XYZ points that consist the FOV surfaces, we can
    # construct the sensor FOV as a set of 3D points.
    out = np.concatenate((fronts, left_side, right_side, top_side, bot_side), axis=0)
    points.append(out) # Multisensor configuration
    
    tStop = perf_counter()
    print(f"FOV point generation took {(tStop-tStart):.2f}s.")
  
  return points

def obtain_trajectory(args: argparse.Namespace) -> Trajectory:
  """Obtains the trajectory from a csv file.

  Args:
      args (argparse.Namespace): Parsed command-line arguments.
      You may regenerate a trajectory if you wish if you set the
      --regenerate flag to True via the command line.
        - You may also input a road section while regenerating the
          points with the --input flag.
          
      You may also use a pregenerated trajectory via the command line
      if you set the --trajectory flag to the path to the trajectory folder.

  Returns:
      trajectory (Trajectory): Container class containing the observer points,
      forwards, leftwards, and upwards vectors.
  """
  print("\nObtaining trajectory!")
  
  if args.regenerate == False:
    # Use existing trajectory
    if args.trajectory == None:
      # Manually open trajectory folder
      Tk().withdraw();
      trajectory_folderpath = tk.filedialog.askdirectory(
        initialdir = ROOT2,
        title = "Please select the trajectory folder"
      )
      print(f"You have chosen to open the trajectory folder:\n{trajectory_folderpath}")   
    else:
      # Get trajectory folder via command line
      trajectory_folderpath = args.trajectory
      print(f"Using pregenerated trajectory folder:\n{os.path.join(ROOT2, trajectory_folderpath)}")
      pass
    
    tStart = perf_counter()
    
    trajectory_files = [
      path for path in os.listdir(trajectory_folderpath) 
      if os.path.isfile(os.path.join(trajectory_folderpath, path))
      ]
    if len(trajectory_files) != 5:
      raise(RuntimeError(f"Trajectory folder is missing files!\nExpected count: 5 (got {len(trajectory_files)})!"))
  
    # Read the csv files and store them into the function
    trajectory_data = dict()
    
    for csv in trajectory_files:
      csv_noext = os.path.splitext(csv)[0]
      path_to_csv = os.path.join(trajectory_folderpath, csv)
      data = np.genfromtxt(path_to_csv, delimiter=",")
      trajectory_data[csv_noext] = data


    observer_points = trajectory_data["observer_points"]
    forwards        = trajectory_data["forwards"]
    leftwards       = trajectory_data["leftwards"]
    upwards         = trajectory_data["upwards"]
    
    tStop = perf_counter()

    print(f"Loading took {(tStop-tStart):.2f}s.")

  else:
    # MANUALLY GENERATE TRAJECTORY
    # Manually obtain file via UI
    las_struct = gen_traj.open_las(verbose=True, args=args)
    traj_config = gen_traj.TrajectoryConfig(floor_box_edge=2.0, point_density=1.0)
    road_points, forwards, leftwards, upwards = gen_traj.generate_trajectory(
      verbose=True, 
      las_obj=las_struct, 
      traj=traj_config
      )
    
    # TODO Move observer point generation to gen_traj.generate_trajectory()
    observer_points = road_points + args.observer_height*upwards
    
  print(f"\nGenerating output using the following parameters:\nobserver_height: {args.observer_height}\nobserver_point:  {args.observer_point}")
  print("(for more information, type 'python sensorpoints.py -h')")
  
  # Finally store our trajectory values into our object
  trajectory = Trajectory(
    observer_points=observer_points,
    forwards=forwards,
    upwards=upwards,
    leftwards=leftwards
    )
  
  return trajectory

# Converted from make_sensor_las_file.m
def align_sensor_points(fov_points: list, trajectory: Trajectory, observer_point: int) -> list:
  """Aligns sensor points to the vehicle's orientation and position 
  at the provided scene number. Output will be in global coordinates
  such that it can be easily superimposed onto the road section itself.

  Args:
      fov_points (list): List containing the XYZ points that make up the sensor
      FOV, for each sensor.
      trajectory (Trajectory): Container class containing the trajectory parameters.
      observer_point (int): Observer point detailing the scene at where
      FOV points should be translated

  Returns:
      transformed_points (list): List containing the XYZ points that make up the sensor
      FOV, for each sensor after our transformation.
  """
  
  # In case if the user does not input a flag for the observer point
  total_road_points = trajectory.getObserverPoints().shape[0]
  if observer_point == None:
    observer_point = int(input(f"Enter the observer point (from 0 to {total_road_points}): "))
  if ((observer_point > total_road_points) or (observer_point < 0)):
    raise ValueError("Observer point is out of range!")


  print("\nAligning FOV points!")
  tStart = perf_counter()
  
  # Rotation matrices are formed as this in the first two dimensions:
  # (note that this is a 2D matrix, the third dimension is the ith rotation matrix)
  # [ fx_i fy_i fz_i ]
  # [ lx_i ly_i lz_i ]
  # [ ux_i uy_i uz_i ]
  #
  # For a point given by [x y z] (row vector):
  # [x y z]*R will take our points from RELATIVE to GLOBAL coordiantes
  # [x y z]*R' (R transposed) will take our points from GLOBAL to RELATIVE coordinates
  
  # Obtain rotation matrices
  # Equivalent to the implementation in MATLAB
  rotation_matrices = np.reshape(
    np.hstack((trajectory.getForwards(), trajectory.getLeftwards(), trajectory.getUpwards())),
    (trajectory.getObserverPoints().shape[0], 3, 3),
    order='F'
    )
  
  rotation_matrices = np.transpose(rotation_matrices, (2,1,0))
  
  # Now we will translate our FOV points to the observer point and align it with the trajectory
  # Also equivalent to the implementation in MATLAB
  transformed_points = []
  for sensorpoints in fov_points:
    out = (
      np.matmul(sensorpoints[(sensorpoints[:,2] > -1.8), :], rotation_matrices[:,:,observer_point])
      + 
      trajectory.getObserverPoints()[observer_point, :]
      )
    transformed_points.append(out)
  
  tStop = perf_counter()
  print(f"FOV point alignment took {(tStop-tStart):.2f}s.")
  
  return transformed_points

def points_to_las(all_points: list, cfg: SensorConfig, args: argparse.Namespace) -> None:
  """Writes our XYZ points to a .las file.

  Args:
      all_points (list): List containing the XYZ points that make up the sensor
      FOV, for each sensor.
      cfg (SensorConfig): Container class for the sensor configuration parameters.
      args (argparse.Namespace): Parsed command-line arguments.
  """
  
  print("\nWriting sensor FOVs to .las!")
  tStart = perf_counter()
  
  # Create output folder
  outpath = ROOT2 / "sensor_fovs"
  
  # New output directory, create directory and gitignore
  if not os.path.exists(outpath):
    os.makedirs(outpath);

    gitignore_path = outpath / '.gitignore'
    with open(gitignore_path, 'w') as f:
      f.write("*.las")
      f.close()

  # 16-bit RGB values
  red = 65535
  green = 65535
  blue = 65535
  
  for i, sensorpoints in enumerate(all_points):
    # Create a las for every sensor
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    
    las.x = sensorpoints[:,0]
    las.y = sensorpoints[:,1]
    las.z = sensorpoints[:,2]
    las.r = np.ones((sensorpoints.shape[0], 1))*red
    las.g = np.ones((sensorpoints.shape[0], 1))*green
    las.b = np.ones((sensorpoints.shape[0], 1))*blue
    
    # outpath_sensor = outpath / os.path.splitext(cfg.sensor_config_filename)[0]
    outpath_sensor = f"{outpath}/{os.path.splitext(cfg.sensor_config_filename)[0]}_frame{args.observer_point}"
    if not os.path.exists(outpath_sensor):
      os.makedirs(outpath_sensor);

    filename = f"sensor_{i}.las"
    
    las.write(os.path.join(outpath_sensor, filename))
    print(f"{filename} was written to\n{outpath_sensor}")
    
  tStop = perf_counter()
  print(f"Writing took {(tStop-tStart):.2f}s.")
  
  return

def parse_cmdline_args() -> argparse.Namespace:
  # use argparse to parse arguments from the command line
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--config", type=str, default=None, help="Path to sensor config file")
  parser.add_argument("--observer_point", type=int, default=None, help="Observer point to place FOV")
  parser.add_argument("--trajectory", type=str, default=None, help="Path to trajectory folder")
  parser.add_argument("--regenerate", type=bool, default=False, help="Manually generate trajectory files again if 'True' (otherwise False)")
  parser.add_argument("--input", type=str, default=None, help="Path to las point cloud (not necessary if you are using existing trajectory)")
  parser.add_argument("--observer_height", type=float, default=1.8, help="Height of the observer in m")
  
  return parser.parse_args()

def main():
  args = parse_cmdline_args()

  config = open_sensor_config_file(args)
  trajectory = obtain_trajectory(args)
  fov_points = generate_sensor_points(config)
  aligned_fov_points = align_sensor_points(fov_points, trajectory, args.observer_point)
  points_to_las(aligned_fov_points, config, args)

  return

if __name__ == "__main__":
  main()