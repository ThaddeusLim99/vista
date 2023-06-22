import numpy as np
import laspy


import sys, os
import argparse
from pathlib import Path
from time import perf_counter

"""
Scenery removal
Eric Cheng
2023-05-25

Trims the input .las point cloud in the y-direction
along the trajectory's road points, with variable range
in the y-direction (assumed constant for now).

Uses the legacy code for using the trajectory to transform
a set of XYZ points, but translated into Python.
"""

# Global variables for file I/O
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
ROOT2 = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import file_tools

# Generate bounding boxes for every road point, with a specified width and length
# at every road point
def generate_bounds(traj: file_tools.Trajectory) -> np.ndarray:
    # TODO Utilise variable widths (and maybe lengths) later on.
    # For now the sizes of each bounding box are assumed constant.
    num_points = traj.getNumPoints()
    
    # Values are indexed from [low, high]
    # I really should make an object or something for this to make things easier to read
    width = np.array([-5, 5]).reshape((1, 2))    # In the y-direction # NOTE temporary: -1.5 to 5.3
    widths = np.repeat(width, num_points, axis=0)    # Repeats the widths
    
    length = np.array([-1.75, 1.75]).reshape((1, 2))   # In the x-direction
    lengths = np.repeat(length, num_points, axis=0)  # Repeats the lengths

    height = np.array([-2, 0.25]).reshape((1, 2))     # In the z-direction 
    heights = np.repeat(height, num_points, axis=0)  # Repeats the heights


    return widths, lengths, heights


def generate_bounding_boxes(
    traj: file_tools.Trajectory, widths: np.ndarray, lengths: np.ndarray, heights: np.ndarray
) -> np.ndarray:
    """Generates bounding box ABCD of a specified size centered and rotated along 
    each and every road point.

    Args:
        traj (file_tools.Trajectory): The trajectory to rotate the bounding boxes with.
        widths (np.ndarray): The widths of each segment, given in an (N, 2) array,
        in the format:
        [[y_min0, y_max0],
         [y_min0, y_max0],
         ...
         [y_minN, y_maxN]]
        
        lengths (np.ndarray): The lengths of each segment, given in an (N, 2) array,
        in the format:
        [[x_min0, x_max0],
         [x_min0, x_max0],
         ...
         [x_minN, x_maxN]]

        heights (np.ndarray): The heights of each segment, given in an (N, 2) array,
        in the format:
        [[z_min0, z_max0],
         [z_min0, z_max0],
         ...
         [z_minN, z_maxN]]

    Returns:
        bounds_transformed (np.ndarray): (N, 8, 3) array containing all of the
        XYZ coordinates for each bounding box, rotated and centered at each road 
        point of the trajectory.
    """

    # bounds_xyz = np.zeros((traj.getNumPoints(), 4, 3), dtype=np.float32)
    bounds_xyz = np.zeros((traj.getNumPoints(), 8, 3), dtype=np.float32)
    
    # Each row A B C D E F G H represents the XYZ point of corners of cuboid ABCDEFGH's corners:
    # A         B     ^x
    #   E         F     >y (from the pov of the observer, along the road)
    # C         D
    #   G         H
    #
    # (note that ABCD is above, while EFGH is below)
    #
    # A   B
    #   E   F
    #                 ^y
    #                   >x (Cartesian, bounding boxes are generated with THIS coordinate system)
    # 
    # C   D
    #   G   H
    #
    # (note that ABCD is above, while EFGH is below)
    
    # Format:
    # bounds_xyz[ROADPOINT, ABCDEFGHINDEX_0to7, XYZINDEX_0to2]
    # example: bounds_xyz[:, 0, 0] sets the x-coordinate of all the A points

    # Set x-coordinate of all ABCDEFGH points
    bounds_xyz[:, 0, 0] = lengths[:, 0] # A
    bounds_xyz[:, 1, 0] = lengths[:, 1] # B
    bounds_xyz[:, 2, 0] = lengths[:, 0] # C
    bounds_xyz[:, 3, 0] = lengths[:, 1] # D
    bounds_xyz[:, 4, 0] = lengths[:, 0] # E
    bounds_xyz[:, 5, 0] = lengths[:, 1] # F
    bounds_xyz[:, 6, 0] = lengths[:, 0] # G
    bounds_xyz[:, 7, 0] = lengths[:, 1] # H

    # Set y-coordinate of all ABCDEFGH points
    bounds_xyz[:, 0, 1] = widths[:, 1] # A
    bounds_xyz[:, 1, 1] = widths[:, 1] # B
    bounds_xyz[:, 2, 1] = widths[:, 0] # C 
    bounds_xyz[:, 3, 1] = widths[:, 0] # D
    bounds_xyz[:, 4, 1] = widths[:, 1] # E
    bounds_xyz[:, 5, 1] = widths[:, 1] # F
    bounds_xyz[:, 6, 1] = widths[:, 0] # G 
    bounds_xyz[:, 7, 1] = widths[:, 0] # H
    
    # Set z-coordinate of all ABCDEFGH points
    bounds_xyz[:, 0, 2] = heights[:, 1] # A
    bounds_xyz[:, 1, 2] = heights[:, 1] # B
    bounds_xyz[:, 2, 2] = heights[:, 1] # C
    bounds_xyz[:, 3, 2] = heights[:, 1] # D
    bounds_xyz[:, 4, 2] = heights[:, 0] # E
    bounds_xyz[:, 5, 2] = heights[:, 0] # F
    bounds_xyz[:, 6, 2] = heights[:, 0] # G
    bounds_xyz[:, 7, 2] = heights[:, 0] # H

    
    ## Now we will transform our bounding points along the trajectory
    # Rotate an xyz point by our rotation matrix R to translate it
    # There is probably a way to vectorize this, but I don't feel like doing indexing magic
    
    # Obtain rotation matrices, converted from the legacy MATLAB code
    rotation_matrices = np.reshape(
        np.hstack((traj.getForwards(), traj.getLeftwards(), traj.getUpwards())),
        (traj.getNumPoints(), 3, 3),
        order="F",
    )
    
    rotation_matrices = np.transpose(rotation_matrices, (2,1,0))

    bounds_transformed = []
    #NOTE I think the bound generation and rotation looks fine so far.
    for i in range(traj.getNumPoints()):
        boundpts = bounds_xyz[i,:,:]
        rotmatr_i = rotation_matrices[:,:,i]
        transformed_pts = np.matmul(boundpts, rotmatr_i) + traj.getRoadPoints()[i, :]
        
        bounds_transformed.append(transformed_pts)
    
    return np.asarray(bounds_transformed)

#TODO Fix the criteria such that these edges do not look jagged
def slice_road_from_bound(las_xyz: np.ndarray, bound_abcdefgh: np.ndarray) -> np.ndarray:
    """Slices the entire road from an individual bounding box, returning
    the indices of any query points that are within the bounding box.

    Args:
        las_x (np.ndarray): The x-coordinates of all the points in the input
        point cloud.
        las_y (np.ndarray): The y-coordinates of all the points in the input
        point cloud.
        bound_abcdefgh (np.ndarray): (8, 3) array of XY points that define the
        bounding box:
        [[Ax, Ay, Az], (top rectangle)
         [Bx, By, Bz],
         [Cx, Cy, Cz],
         [Dx, Dy, Dz],
         [Ex, Ey, Ez], (bottom rectangle)
         [Fx, Fy, Fz],
         [Gx, Gy, Gz],
         [Hx, Hy, Hz]]
         
        This is what each bounding box looks like:
        A         B     ^x
          E         F     >y (from the pov of the observer, along the road)
        C         D
          G         H

    Returns:
        idx (np.ndarray): Indices of the .las point cloud that are in
        the bounding box.
    """
    
    # Find the three perpendicular directions of bounding cuboid ABCDEFGH.
    # In this case:
    # i (AB), j (AC), k (AE)
    #
    # A point x lies within the box when these three conditions are satisfied:

    
    # Determine whether or not some arbitary point(s) p is within a cuboid ABCDEFGH
    # Perpendicular edges of the rectangular box (see the diagram above...)
    i = bound_abcdefgh[1] - bound_abcdefgh[0] # i = AB = [Bx-Ax, By-Ay, Bz-Az]
    j = bound_abcdefgh[2] - bound_abcdefgh[0] # j = AC = [Cx-Ax, Cy-Ay, Cz-Az]
    k = bound_abcdefgh[4] - bound_abcdefgh[0] # k = AE = [Ex-Ax, Ey-Ay, Ez-Az]

    p = las_xyz - bound_abcdefgh[0]           # p = Ap = [px-ax, py-ay, pz-az]
    
    con0 = (np.dot(p, i) >= 0) & (np.dot(p, i) <= np.dot(i, i))
    con1 = (np.dot(p, j) >= 0) & (np.dot(p, j) <= np.dot(j, j))
    con2 = (np.dot(p, k) >= 0) & (np.dot(p, k) <= np.dot(k, k))

    idx = np.where(con0 & con1 & con2)[0]
    
    return idx

def remove_scenery(bounds: np.ndarray, las: laspy.LasData, las_filename: str) -> None:
    """Removes points from a point cloud that are outside 
    a defined range, where the indices of any points from the 
    .las file that are within of a bound at a road point are 
    stored, and then indexed from the input point cloud itself
    to obtain the trimmed road section.

    Args:
        bounds (np.ndarray): (N, 8, 3) array of XYZ points that defines each of the
        bounds at a specific road point, where N is the number of road points.
        las (laspy.LasData): Our input point cloud.
        las_filename (str): The filename of out input point cloud.
    """
    
    las_xyz = np.array([las.x, las.y, las.z], dtype=np.float32).T
    
    '''
    import multiprocessing as mp
    from tqdm import tqdm
    
    mp.freeze_support()
    cores = mp.cpu_count() - 1
    total_points = bounds.shape[0]
    
    # Define the arguments that will be called upon in parallel
    slice_args = [(las_x, las_y, bounds[frame, :, 0:2]) for frame in range(total_points)]
    
    tStart = perf_counter()
    print("")
    with mp.Pool(cores) as p:  # Opening up more process pools
        indices = p.starmap(
            slice_road_from_bound,
            tqdm(
                slice_args,
                total=total_points,
                desc="Slicing point cloud"
            )
        )

        p.close()  # No new tasks for our pool
        p.join()  # Wait for all processes to finish 
        
    tStop = perf_counter()
    print(f"Multiprocessing took {tStop-tStart:.2f}s")
    '''
    
    # Running multiprocessing with joblib for speed.
    import joblib
    from joblib import Parallel, delayed
    from tqdm import tqdm
    
    total_points = bounds.shape[0]
    cores = (joblib.cpu_count() - 1)

    # Define the arguments that will be called upon in parallel
    slice_args = [(las_xyz, bounds[frame, :, :]) for frame in range(total_points)]
    
    tStart = perf_counter()
    print("")
    indices = Parallel(n_jobs=cores, backend='loky')( # Switched to loky backend to maybe suppress errors?
        delayed(slice_road_from_bound)(arg_las_xyz, arg_bounds)
        for arg_las_xyz, arg_bounds in tqdm(slice_args, 
                                            total=total_points, 
                                            desc=f"Slicing point cloud")
        )
    tStop = perf_counter()
    print(f"\nRemoving overlapped points...")
    
    # We don't care about overlapped bounding boxes
    indices = np.unique(np.concatenate(indices))
    print(f"Successfully removed {las.x.shape[0]-indices.shape[0]} points (down from {las.x.shape[0]}) in {tStop-tStart:.2f}s.")
    
    # Now we can remove the non-road points
    trimmed_points = las.points[indices]
    
    # Define our output directory
    output_folder = os.path.join(ROOT2, "trimmed")
    if not os.path.exists(output_folder):
      os.makedirs(output_folder) 
    
    # Write our trimmed point cloud
    las_filename = os.path.splitext(las_filename)[0]
    
    header = las.header # Retain offsets, data format, etc...
    output_folderpath2file = os.path.join(output_folder, f"{las_filename}_y_trimmed.las")  
    with laspy.open(output_folderpath2file, mode='w', header=header) as writer:
        writer.write_points(trimmed_points)
        print(f"\nTrimmed point cloud has been successfully written to {output_folderpath2file}")
    
    return

# NOTE PointFormat will be 1 for our input point clouds.
def open_las(args: argparse.Namespace) -> laspy.LasData:
    
    import tkinter as tk
    from tkinter import Tk
    
    try:
        arg_input = args.input
    except AttributeError:
        arg_input = None
    
    if arg_input == None:
        # Manually obtain file via UI
        Tk().withdraw()
        las_filename = tk.filedialog.askopenfilename(
            filetypes=[(".las files", "*.las"), ("All files", "*")],
            initialdir=ROOT2,
            title="Please select the main point cloud",
        )

        print(f"You have chosen to open the point cloud:\n{las_filename}")

    else:
        las_filename = args.input

    # Obtain the las file name itself rather than the path for csv output
    las_filename_cut = os.path.basename(las_filename)

    # Note: lowercase dimensions with laspy give the scaled value
    raw_las = laspy.read(las_filename)

    return raw_las, las_filename_cut

# writes the XYZ bounding points for visualization in cloudcompare
def debug_write_bounds(bounds: np.ndarray, las_filename: str):
    # bounds is given as a (road_pts, 4, 3) array
    # bounds = bounds.reshape(((bounds.shape[0]*8), 3), order='F')
    las_filename = os.path.splitext(las_filename)[0]
    
    output_folder = os.path.join(ROOT2, "trimmed")
    if not os.path.exists(output_folder):
      os.makedirs(output_folder) 
    
    # np.savetxt(f"{output_folder}/{las_filename}_boundingpoints.csv", bounds[:-1:20, :], delimiter=',')
    np.savetxt(f"{output_folder}/{las_filename}_boundingpoints.csv", bounds[50, :, :], delimiter=',')
    # np.savetxt(f"{output_folder}/{las_filename}_boundingpoints.csv", bounds, delimiter=',')
    print(f"Debug bounding points written to {output_folder}/{las_filename}")

    return

# Driver function for everything
def main():
    args = file_tools.parse_cmdline_args()
    traj = file_tools.obtain_trajectory_details(args)

    widths, lengths, heights = generate_bounds(traj)
    bounds = generate_bounding_boxes(traj, widths, lengths, heights)
    
    las, las_filename = open_las(args)
    # remove_scenery(bounds, las, las_filename)
    
    # For testing
    debug_write_bounds(bounds, las_filename)
    return


if __name__ == "__main__":
    main()