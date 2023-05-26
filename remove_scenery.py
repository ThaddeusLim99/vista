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
    width = np.array([-20, 20]).reshape((1, 2))  # In the y-direction
    widths = np.repeat(width, num_points, axis=0)

    length = np.array([-1.5, 1.5]).reshape((1, 2))  # In the x-direction
    lengths = np.repeat(length, num_points, axis=0)

    return widths, lengths

def generate_bounding_boxes(
    traj: file_tools.Trajectory, widths: np.ndarray, lengths: np.ndarray
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
         
        The z-coordinate here is zero for all bounding points.
        
        lengths (np.ndarray): The lengths of each segment, given in an (N, 2) array,
        in the format:
        [[x_min0, x_max0],
         [x_min0, x_max0],
         ...
         [x_minN, x_maxN]]
         
        The z-coordinate here is zero for all bounding points.

    Returns:
        bounds_transformed (np.ndarray): (N, 4, 3) array containing all of the
        XYZ coordinates for each bounding box, rotated centered at each road 
        point of the trajectory.
    """

    rotation_matrices = np.reshape(
        np.hstack((traj.getForwards(), traj.getLeftwards(), traj.getUpwards())),
        (traj.getNumPoints(), 3, 3),
        order="F",
    )
    
    rotation_matrices = np.transpose(rotation_matrices, (2,1,0))


    bounds_xyz = np.zeros((traj.getNumPoints(), 4, 3))
    
    # Each row A B C D represents the points of corners of rectangle ABCD's corners:
    # A         B   ^x
    #      M          >y (from the pov of the observer)
    # C         D
    #
    # A   B
    # 
    #               ^y
    #   M             >x (Cartesian, this will be rotated)
    # 
    # 
    # C   D
    
    # Format:
    # bounds_xyz[ROADPOINT, ABCDINDEX_0to3, XYZINDEX_0to2]
    # example: bounds_xyz[:, 0, 0] sets the x-coordinate of all the A points

    # Set x-coordinate of all ABCD points
    bounds_xyz[:, 0, 0] = lengths[:, 0] # A
    bounds_xyz[:, 1, 0] = lengths[:, 0] # B
    bounds_xyz[:, 2, 0] = lengths[:, 1] # C
    bounds_xyz[:, 3, 0] = lengths[:, 1] # D

    # Set y-coordinate of all ABCD points
    bounds_xyz[:, 0, 1] = widths[:, 1] # A
    bounds_xyz[:, 1, 1] = widths[:, 0] # B
    bounds_xyz[:, 2, 1] = widths[:, 1] # C 
    bounds_xyz[:, 3, 1] = widths[:, 0] # D
    
    # We don't care about the z-coordinate, assume that our bounding box
    # is flat (on the plane z=0). We will take the z-coordinate anyways.
    
    # Transform our bounding points to be along the trajectory
    # Rotate an xyz point by our rotation matrix R to translate it
    # There is probably a way to vectorize this, but I don't feel like doing indexing magic
    bounds_transformed = []
    for i in range(traj.getNumPoints()):
        boundpts = bounds_xyz[i,:,:]
        rotmatr_i = rotation_matrices[:,:,i]
        
        # @ is matrix multiplication
        transformed_pts = boundpts @ rotmatr_i + traj.getRoadPoints()[i,:]
        bounds_transformed.append(transformed_pts)
    
    return np.asarray(bounds_transformed)

def slice_road_from_bound(las_x: np.ndarray, las_y: np.ndarray, bound_abcd: np.ndarray) -> np.ndarray:
    """Slices the entire road from an individual bounding box, returning
    the indices of any query points that are within the bounding box.

    Args:
        las_x (np.ndarray): The x-coordinates of all the points in the input
        point cloud.
        las_y (np.ndarray): The y-coordinates of all the points in the input
        point cloud.
        bound_abcd (np.ndarray): (4, 2) array of XY points that define the
        bounding box:
        [[Ax, Ay],
         [Bx, By],
         [Cx, Cy],
         [Dx, Dy]]

    Returns:
        idx (np.ndarray): Indices of the .las point cloud that are in
        the bounding box.
    """
    
    # Here we will do the equivalent of projecting the bounding box and the .las points
    # onto the XY plane by ignoring the z-coordinate. The z-coordinate does not matter 
    # very much here; we will take all points in the bounding box.
    #
    # A         B   ^x
    #      M          >y (from the POV of the observer)
    # C         D

    # Per the vector definition:
    # AM = (Mx-Ax, My-Ay)
    # BM = (Mx-Bx, My-By)
    AM = np.array([las_x            - bound_abcd[0, 0], las_y            - bound_abcd[0, 1]], dtype=np.float32).T
    BM = np.array([las_x            - bound_abcd[1, 0], las_y            - bound_abcd[1, 1]], dtype=np.float32).T
    
    # AB = (Bx-Ax, By-Ay)
    # AC = (Cx-Ax, Cy-Ay)
    AB = np.array([bound_abcd[1, 0] - bound_abcd[0, 0], bound_abcd[1, 1] - bound_abcd[0, 1]])
    AC = np.array([bound_abcd[2, 0] - bound_abcd[0, 0], bound_abcd[2, 1] - bound_abcd[0, 1]])

    # A query point (or points) M is/are in rectangle ABCD iff. the following condition
    # (0 < np.dot(AM, AB) < np.dot(AB, AB)) and (0 < np.dot(BM, AC) < np.dot(AC, AC))
    # is satisfied. Given that two vectors AB, and AC are perpendicular (seen above),
    # we only need to evalulate the projections of the query point M on AB and AC.
    # 
    # M is only inside the rectangle if its projection onto line segments AB and AC are
    # inside BOTH line segments.
    # Source: https://math.stackexchange.com/questions/190111/how-to-check-if-a-point-is-inside-a-rectangle
    
    foo = (np.dot(AM, AB) > 0) & (np.dot(AM, AB) < np.dot(AB, AB))
    bar = (np.dot(BM, AC) > 0) & (np.dot(BM, AC) < np.dot(AC, AC))

    idx = np.where(foo & bar)[0] 
    return idx

def remove_scenery(bounds: np.ndarray, las: laspy.LasData, las_filename: str) -> None:
    """Removes points from a point cloud that are outside 
    a defined range, where the indices of any points from the 
    .las file that are within of a bound at a road point are 
    stored, and then indexed from the input point cloud itself
    to obtain the trimmed road section.

    Args:
        bounds (np.ndarray): (N, 4, 3) array of XYZ points that defines each of the
        bounds at a specific road point, where N is the number of road points.
        las (laspy.LasData): Our input point cloud.
        las_filename (str): The filename of out input point cloud.
    """
    
    # Each one of our processes will have a copy of the las data
    # instead of taking from the las data itself, so that our multiprocessing works
    las_x = las.x
    las_y = las.y
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
    slice_args = [(las_x, las_y, bounds[frame, :, 0:2]) for frame in range(total_points)]
    
    tStart = perf_counter()
    print("")
    indices = Parallel(n_jobs=cores, backend='loky')( # Switched to loky backend to maybe suppress errors?
        delayed(slice_road_from_bound)(arg_las_x, arg_las_y, arg_bounds)
        for arg_las_x, arg_las_y, arg_bounds in tqdm(slice_args, 
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
        print(f"\n{las_filename} has been successfully written to {output_folderpath2file}")
    

    # test visualization of slice, temp?
    '''
    ### test
    import open3d as o3d
    
    print(las.x.shape)
    indices = np.unique(np.concatenate(indices))
    print(f"Successfully removed {las.x.shape[0]-indices.shape[0]} points.")


    temp_points = np.vstack((las.x[indices], las.y[indices], las.z[indices])).T
    pcd = o3d.geometry.PointCloud()

    import matplotlib
    las_intensity = las.intensity[indices]
    
    # Normalize intensity values to [0, 1], then assign RGB values
    normalizer = matplotlib.colors.Normalize(np.min(las_intensity), np.max(las_intensity))
    las_rgb = matplotlib.cm.gray(normalizer(las_intensity))[:,:-1]
    pcd.colors = o3d.utility.Vector3dVector(las_rgb) # cmap(las_intensity) returns RGBA, cut alpha channel
    pcd.points = o3d.utility.Vector3dVector(temp_points)
    
    
    ## test end
    o3d.visualization.draw_geometries([pcd])
    '''
    
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
def debug_write_bounds(bounds: np.ndarray):
    # bounds is given as a (road_pts, 4, 3) array
    bounds = bounds.reshape(((bounds.shape[0]*4), 3), order='F')
    print(bounds.shape)
    
    np.savetxt(f"{ROOT2}/boundingpoints.csv", bounds, delimiter=',')

    return

# Driver function
def main():
    args = file_tools.parse_cmdline_args()
    traj = file_tools.obtain_trajectory_details(args)

    widths, lengths = generate_bounds(traj)
    bounds = generate_bounding_boxes(traj, widths, lengths)
    
    las, las_filename = open_las(args)
    remove_scenery(bounds, las, las_filename)
    
    # Test
    # debug_write_bounds(bounds)
    return


if __name__ == "__main__":
    main()
