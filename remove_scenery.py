import numpy as np
import laspy


import sys, os
import argparse
from pathlib import Path
from time import perf_counter


# Global variables for file I/O
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
ROOT2 = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


import file_tools

# Cut points that are outside the curve of rectangles in a hardcoded range in the
# y-direction... (we can automatically determine the range later)
# IMPLEMENTATION
# -BOUNDING RECTANGLES
# Using the trajectory, and the road points, we can fit a rectangle along the xy-plane
# centered at each road point where we will sample road points that fit within all of the rectangles.
#  - At each rectangle we take points regardless of z-direction.
#
# -EXTRACTION
# Store unique indices of all points that are inside of our bounding rectangles,
# and then get our trimmed point cloud. Write to las.


# Generate widths in the y-direction for each road point
def generate_bounds(traj: file_tools.Trajectory) -> np.ndarray:
    # TODO Utilise variable widths (and maybe lengths) later on.
    num_points = traj.getNumPoints()
    width = np.array([-25, 25]).reshape((1, 2))  # In the y-direction
    widths = np.repeat(width, num_points, axis=0)

    length = np.array([-1.5, 1.5]).reshape((1, 2))  # In the x-direction
    lengths = np.repeat(length, num_points, axis=0)

    return widths, lengths


def generate_bounding_boxes(
    traj: file_tools.Trajectory, widths: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    # Given an origin point, obtain the coordinates of four coordinates about it.
    # The coordinates of our four corner points will be determined by widths. For
    # now it will be constant.
    # Using the trajectory, rotate these corner points. These are our bounding
    # rectangles, where we want points that are in the bounding rectangles.

    rotation_matrices = np.reshape(
        np.hstack((traj.getForwards(), traj.getLeftwards(), traj.getUpwards())),
        (traj.getNumPoints(), 3, 3),
        order="F",
    )

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
    bounds_xyz = np.zeros((traj.getNumPoints(), 4, 3))
    # bounds_xyz[:, ABCDINDEX, XYZINDEX]
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
    
    # We don't care about the z-coordinate, assume zero

    # Transform our bounds now
    bounds_transformed = (
        np.matmul(bounds_xyz[:,:,:], rotation_matrices[:,:,:])
        +
        np.expand_dims(traj.getObserverPoints(), axis=1)
        ) 

    # print(bounds_transformed[0,:,:])
    
    # example: 
    # xy coordinates of all 4001 A points
    # A = bounds_transformed[:,0,0:2]
    # print(A)
    
    return bounds_transformed

def slice_road_from_bound(las_x: np.ndarray, las_y: np.ndarray, bound_abcd: np.ndarray):
    
    AM = np.array([las_x            - bound_abcd[0, 0], las_y            - bound_abcd[0, 1]], dtype=np.float32).T
    BM = np.array([las_x            - bound_abcd[1, 0], las_y            - bound_abcd[1, 1]], dtype=np.float32).T
    
    AB = np.array([bound_abcd[1, 0] - bound_abcd[0, 0], bound_abcd[1, 1] - bound_abcd[0, 1]])
    AC = np.array([bound_abcd[2, 0] - bound_abcd[0, 0], bound_abcd[2, 1] - bound_abcd[0, 1]])

    # (0 < np.dot(am, ab) < np.dot(ab, ab)) and (0 < np.dot(am, ad) < np.dot(ad, ad))
    foo = (np.dot(AM, AB) > 0) & (np.dot(AM, AB) < np.dot(AB, AB))
    bar = (np.dot(BM, AC) > 0) & (np.dot(BM, AC) < np.dot(AC, AC))

    idx = np.where(foo & bar)[0] 
    return idx

def remove_scenery(bounds: np.ndarray, las: laspy.LasData, las_filename: str):
    # https://math.stackexchange.com/questions/190111/how-to-check-if-a-point-is-inside-a-rectangle
    # Point M (x,y) is inside a rectangle ABCD iff.
    # (0 < AM dot AB < AB dot AB) and (0 < AM dot AD < AD dot AD); dot products of vectors

    # bounds is given by a 4001x4x3 array
    # to access a bound, do bounds[ROADPT, :, :]. this returns a 4x3 array of xyz points
    # rows are the xyz coordinates of the rectangle (ABCD in order)
    
    # Get indices of points in the las file that are within our bounding box
    # Grab only the points from a file which XY 
    # coordinates are within all of our bounding boxes
    
    #las_x = las.x
    #las_y = las.y
    
    '''
    import multiprocessing as mp
    from tqdm import tqdm
    
    mp.freeze_support()
    cores = mp.cpu_count() - 1
    total_points = bounds.shape[0]
    slice_args = [(las.x, las.y, bounds[frame, :, 0:2]) for frame in range(total_points)]
    
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
    '''
    

    import joblib
    from joblib import Parallel, delayed
    from tqdm import tqdm
    
    total_points = bounds.shape[0]
    cores = (joblib.cpu_count() - 1)

    # Define the arguments that will be ran upon in parallel
    slice_args = [(las.x, las.y, bounds[frame, :, 0:2]) for frame in range(total_points)]
    
    indices = Parallel(n_jobs=cores, backend='loky')( # Switched to loky backend to maybe suppress errors?
        delayed(slice_road_from_bound)(arg_las_x, arg_las_y, arg_bounds)
        for arg_las_x, arg_las_y, arg_bounds in tqdm(slice_args, 
                                                     total=total_points, 
                                                     desc=f"Slicing point cloud")
        )

    indices = np.unique(np.concatenate(indices))
    print(f"Successfully removed {las.x.shape[0]-indices.shape[0]} points (down from {las.x.shape[0]}).")
    
    trimmed_points = las.points[indices]
    
    # header = laspy.LasHeader(point_format=1, version="1.2")
    # header.offsets = np.array([0, -5000000, 0])
    header = las.header
    with laspy.open("test.las", mode='w', header=header) as writer:
        writer.write_points(trimmed_points)
    
    
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
    
    '''
    # TODO Parallelize this loop.
    for point, bound_abcd in enumerate(bounds[:,:,0:2]):
        # ((0 < np.dot(am, ab) < np.dot(ab, ab)) and (0 < np.dot(am, ad) < np.dot(ad, ad)))
        # Vector from point of the rectangle to arbitary points M, in the 2D sense

        # [ABCD, XY]
        # Mathematical vector definition
        # A         B   ^x
        #      M          >y (from the pov of the observer)
        # C         D
        
        # AB and AC are perpendicular.
        
        # SO sample, they took AB and BC as perpendicular.
        # 0 <= dot(AB,AM) <= dot(AB,AB) 
        # &&
        # 0 <= dot(BC,BM) <= dot(BC,BC)
        
        # in our case, we take AB and AC as perpendicular.
        # 0 <= dot(AB,AM) <= dot(AB,AB) 
        # &&
        # 0 <= dot(AC,BM) <= dot(AC,AC)
        
        # Given the fact that our bounding rectangle is 
        # represented by four XY points A, B, C, D, 
        # with AB and AC being perpendicular (see below),
        #
        # A         B   ^x
        #      M          >y (from the pov of the observer)
        # C         D
        # 
        # we only need to check the projections of query point
        # M on AB and AC to identify if M is inside ABCD.
       
        
        AM = np.array([las_x            - bound_abcd[0, 0], las_y            - bound_abcd[0, 1]], dtype=np.float32).T
        BM = np.array([las_x            - bound_abcd[1, 0], las_y            - bound_abcd[1, 1]], dtype=np.float32).T
        
        AB = np.array([bound_abcd[1, 0] - bound_abcd[0, 0], bound_abcd[1, 1] - bound_abcd[0, 1]])
        AC = np.array([bound_abcd[2, 0] - bound_abcd[0, 0], bound_abcd[2, 1] - bound_abcd[0, 1]])

        # (0 < np.dot(am, ab) < np.dot(ab, ab)) and (0 < np.dot(am, ad) < np.dot(ad, ad))
        foo = (np.dot(AM, AB) > 0) & (np.dot(AM, AB) < np.dot(AB, AB))
        bar = (np.dot(BM, AC) > 0) & (np.dot(BM, AC) < np.dot(AC, AC))

        idx = np.where(foo & bar)[0]
    '''

    return

# NOTE PointFormat will be one for our input point clouds.
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


def debugfunc(bounds: np.ndarray):
    # import pandas as pd
    for point, bound_abcd in enumerate(bounds[:,:,:]):
        np.savetxt(f"test/{point}.csv", bound_abcd, delimiter=',')

    return

def main():
    args = file_tools.parse_cmdline_args()
    traj = file_tools.obtain_trajectory_details(args)

    widths, lengths = generate_bounds(traj)
    bounds = generate_bounding_boxes(traj, widths, lengths)
    
    debugfunc(bounds)
    #las, las_filename = open_las(args)
    #remove_scenery(bounds, las, las_filename)
    
    return


if __name__ == "__main__":
    main()
