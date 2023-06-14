import json
import os
import math
import re
import glob
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

import file_tools

#gets the first number in a string (read left to right)
#used to get the frame number out of the filename
#assumes that the frame number is the first number in the filename
def get_first_number(substring: str):
    numbers = re.findall(r'\d+', substring)
    if numbers:
        return int(numbers[0])
    else:
        return None

#gets the substring from the end of a string till the first forward
#or backwards slash encountered
def get_folder(string: str):
    match = re.search(r"([\\/])([^\\/]+)$", string)
    if match:
        result = match.group(2)
        return result
    return None

def multiprocessed_vol_funct(input_tuple: tuple):
    """Perfroms the volumetric method for a single scene.
    Called in the parallelized for loop.

    Args:
        input_tuple (tuple): Our input tuple of the form
        (voxel_rsize, voxel_asize, voxel_esize, data, i, vistaoutput_path, point_density, max_volume), where:
         - voxel_rsize: Range precision in meters
         - voxel_asize: Azimuth precision in meters
         - voxel_esize: Elevation precision in meters
         - data: Our sensor configuration
         - point_density: Density of the points in points per meter
         - max_volume: The max possible volume for all the voxels

    Returns:
        i (int): The frame number
        output (list): The ratio of occupied volume to total volume.
    """
    voxel_rsize, voxel_asize, voxel_esize, data, i, vistaoutput_path, point_density, max_volume = input_tuple
    filename = "output_" + str(i) + "_0.11.txt"
    f = os.path.join(vistaoutput_path, filename)   
    file = np.loadtxt(f, delimiter=',',skiprows=1)
    numFrame = get_first_number(filename)
    pc = np.divide(file,1000) # convert from mm to meters
    inputExtended = (pc,voxel_rsize, voxel_asize, voxel_esize, data,i)

    occ = occupancy_volume(inputExtended)/max_volume

    return i,[((numFrame) * point_density), occ]


def multiprocessed_count_funct(input_tuple: tuple):
    """Perfroms the simple method for a single scene.
    Called in the parallelized for loop.

    Args:
        input_tuple (tuple): Our input tuple of the form
        (voxel_rsize, voxel_asize, voxel_esize, data, i, vistaoutput_path, point_density), where:
         - voxel_rsize: Range precision in meters
         - voxel_asize: Azimuth precision in meters
         - voxel_esize: Elevation precision in meters
         - data: Our sensor configuration
         - point_density: Density of the points in points per meter

    Returns:
        i (int): The frame number
        output (list): The ratio of occupied volume to total volume.
    """
    voxel_rsize, voxel_asize, voxel_esize, data, i, vistaoutput_path, point_density = input_tuple
    filename = "output_" + str(i) + "_0.11.txt"
    f = os.path.join(vistaoutput_path, filename)
    file = np.loadtxt(f, delimiter=',',skiprows=1)
    numFrame = get_first_number(filename)
    pc = np.divide(file,1000) # convert from mm to meters
    inputExtended = (pc,voxel_rsize, voxel_asize, voxel_esize, data,i)

    return i,[((numFrame) * point_density), occupancy_count(inputExtended)]

#https://github.com/numpy/numpy/issues/5228
#http://matlab.izmiran.ru/help/techdoc/ref/cart2sph.html#:~:text=cart2sph%20(MATLAB%20Functions)&text=Description-,%5BTHETA%2CPHI%2CR%5D%20%3D%20cart2sph(X%2C,and%20Z%20into%20spherical%20coordinates.
def cart2sph(x, y, z) -> np.ndarray:
    # Converted from matlab's cart2sph() function.
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

#(pc,voxel_rsize, voxel_asize, voxel_esize, data, i)
def occupancy_volume(input: tuple) -> np.float32:
    """Computes the volume of all occupied voxels,
    with working spherical coordinates.

    Args:
        input (tuple): Tuple of form (pc, r_size, azimuth_size, elevation_size, sensorcon).
         - pc: The point cloud to voxelize.
         - r_size: Voxel size of the spherical coordinate; radius of sphere.
         - azimuth_size: Azimuth angle of each voxel from the center
         - elevation_size: Elevation angle of each voxel from the center
         - sensorcon: Our sensor configuration.

    Returns:
        total_volume (np.float32): The volume of all of occupied voxels in the scene.
    """

    # Working with the Vista output point cloud
    
    pc = input[0]
    r_size = input[1]
    azimuth_size = input[2]
    elevation_size = input[3]
    sensorcon = input[4]
    
    # Filter out the x y z components
    xyz_point_cloud_data = pc[:, 0:3]
        
    ## Converting the point cloud into spherical coordinates
    azimuth, elevation, r = cart2sph(xyz_point_cloud_data[:, 0], xyz_point_cloud_data[:, 1], xyz_point_cloud_data[:, 2])
    
    # Converting from radians to Degrees
    # For both azimuth and elevation, both the min are now zero, and the max
    # has been changed from 180 to 360 for azimuth, and from 90 to 180 for
    # elevation. We basically got rid of negative angle and shifted the
    # notation by the respectable amount.
    azimuth = np.rad2deg(azimuth) + 180
    azimuth = np.mod(azimuth, 360)
    elevation = np.rad2deg(elevation) + 90
    elevation = np.mod(elevation, 180)
    spherical_point_cloud_data = np.transpose(np.array([r, elevation, azimuth]))
    
    # Origin coordinate setting. Where the sensor is. Move to outside the
    # function in the future.
    voxel_struct = {}
    voxel_struct["r_size"] = r_size
    voxel_struct["a_size"] = azimuth_size
    voxel_struct["e_size"] = elevation_size
    
    ## Removing values that are over the constraints
    # Range
    spherical_point_cloud_data = spherical_point_cloud_data[spherical_point_cloud_data[:, 0] < sensorcon["r_high"]]
    spherical_point_cloud_data = spherical_point_cloud_data[spherical_point_cloud_data[:, 0] >= sensorcon["r_low"]] 
     
    # Elevation range
    spherical_point_cloud_data = spherical_point_cloud_data[spherical_point_cloud_data[:, 1] < sensorcon["e_high"]]
    spherical_point_cloud_data = spherical_point_cloud_data[spherical_point_cloud_data[:, 1] >= sensorcon["e_low"]]
    # Azimuth range
    spherical_point_cloud_data = spherical_point_cloud_data[spherical_point_cloud_data[:, 2] < sensorcon["a_high"]]
    spherical_point_cloud_data = spherical_point_cloud_data[spherical_point_cloud_data[:, 2] >= sensorcon["a_low"]]
    ## Spherical voxelization
    spherical_point_cloud_data[:, 2] = np.floor(spherical_point_cloud_data[:, 2]/voxel_struct["a_size"])
    spherical_point_cloud_data[:, 1] = np.floor(spherical_point_cloud_data[:, 1]/voxel_struct["e_size"])
    spherical_point_cloud_data[:, 0] = np.floor(spherical_point_cloud_data[:, 0]/voxel_struct["r_size"])

    ## Only keep unique voxels
    # Removing duplicates creates the voxel data and also sorts it
    # We are also removing occlusions by sorting the original coordinates by
    # the distance from the sensor. Then we run unique on the azimuth and
    # elevation angle. The index output from that will be used to basically get
    # rid of all the voxels behind a certain angle coordinate.
    spherical_point_cloud_data = spherical_point_cloud_data[spherical_point_cloud_data[:, 0].argsort()]
    vx, unique_indices = np.unique(spherical_point_cloud_data, axis=0, return_index=True)
    vx = vx[np.argsort(unique_indices)]
    
    ## Finding volume
    # Getting the range for integration
    a_low = vx[:, 2] * voxel_struct["a_size"]
    e_low = vx[:, 1] * voxel_struct["e_size"]
    r_low = vx[:, 0] * voxel_struct["r_size"]
    a_high = a_low + voxel_struct["a_size"]
    e_high = e_low + voxel_struct["e_size"]
    r_high = r_low + voxel_struct["r_size"]
    
    volume = (1/3)*((np.power(r_high,3)) - (np.power(r_low,3)))\
        * (np.cos(np.deg2rad(e_low)) - np.cos(np.deg2rad(e_high)))\
        * (np.deg2rad(a_high) - np.deg2rad(a_low))
    
    # Just sum the volume matrix
    total_volume = sum(volume)
    
    return total_volume   

#(pc,voxel_rsize, voxel_asize, voxel_esize, data,i)
def occupancy_count(input: tuple) -> np.float32:
    """Computes the ratio of occupied voxels to the total amount of
    voxels.

    Args:
        input (tuple): Tuple of form (pc, r_size, azimuth_size, elevation_size, sensorcon).
         - pc: The point cloud to voxelize.
         - r_size: Voxel size of the spherical coordinate; radius of sphere.
         - azimuth_size: Azimuth angle of each voxel from the center
         - elevation_size: Elevation angle of each voxel from the center
         - sensorcon: Our sensor configuration.

    Returns:
        out_ratio (np.float32): The ratio of occupied voxels to the the total number.
    """
    
    # Working with the Vista output point cloud
    
    pc = input[0]
    r_size = input[1]
    azimuth_size = input[2]
    elevation_size = input[3]
    sensorcon = input[4]
    
    # Filter out the x y z components
    xyz_point_cloud_data = pc[:, 0:3]
        
    # Converting the point cloud into spherical coordinates
    azimuth, elevation, r = cart2sph(xyz_point_cloud_data[:, 0], xyz_point_cloud_data[:, 1], xyz_point_cloud_data[:, 2])
    
    # Converting from radians to Degrees
    # For both azimuth and elevation, both the min are now zero, and the max
    # has been changed from 180 to 360 for azimuth, and from 90 to 180 for
    # elevation. We basically got rid of negative angle and shifted the
    # notation by the respectable amount.
    azimuth = np.rad2deg(azimuth) + 180
    azimuth = np.mod(azimuth, 360)
    elevation = np.rad2deg(elevation) + 90
    elevation = np.mod(elevation, 180)
    spherical_point_cloud_data = np.transpose(np.array([r, elevation, azimuth]))
    
    # Origin coordinate setting. Where the sensor is. Move to outside the
    # function in the future.
    voxel_struct = {}
    voxel_struct["r_size"] = r_size
    voxel_struct["a_size"] = azimuth_size
    voxel_struct["e_size"] = elevation_size    
    
    ## Removing values that are over the constraints
    # Range
    spherical_point_cloud_data = spherical_point_cloud_data[spherical_point_cloud_data[:, 0] < sensorcon["r_high"]]
    spherical_point_cloud_data = spherical_point_cloud_data[spherical_point_cloud_data[:, 0] >= sensorcon["r_low"]] 
    
    # Elevation range
    spherical_point_cloud_data = spherical_point_cloud_data[spherical_point_cloud_data[:, 1] < sensorcon["e_high"]]
    spherical_point_cloud_data = spherical_point_cloud_data[spherical_point_cloud_data[:, 1] >= sensorcon["e_low"]]
    
    # Azimuth range
    spherical_point_cloud_data = spherical_point_cloud_data[spherical_point_cloud_data[:, 2] < sensorcon["a_high"]]
    spherical_point_cloud_data = spherical_point_cloud_data[spherical_point_cloud_data[:, 2] >= sensorcon["a_low"]]

    ## Spherical voxelization
    spherical_point_cloud_data[:, 2] = np.floor(spherical_point_cloud_data[:, 2]/voxel_struct["a_size"])
    spherical_point_cloud_data[:, 1] = np.floor(spherical_point_cloud_data[:, 1]/voxel_struct["e_size"])
    spherical_point_cloud_data[:, 0] = np.floor(spherical_point_cloud_data[:, 0]/voxel_struct["r_size"])
    
    ## Only keep unique voxels
    # Removing duplicates creates the voxel data and also sorts it
    # We are also removing occlusions by sorting the original coordinates by
    # the distance from the sensor. Then we run unique on the azimuth and
    # elevation angle. The index output from that will be used to basically get
    # rid of all the voxels behind a certain angle coordinate.
    spherical_point_cloud_data = spherical_point_cloud_data[spherical_point_cloud_data[:, 0].argsort()]
    vx, unique_indices = np.unique(spherical_point_cloud_data, axis=0, return_index=True)
    vx = vx[np.argsort(unique_indices)]

    ## Finding the total amount of voxels in this range with config
    azimuth_capacity = np.floor((sensorcon["a_high"]-sensorcon["a_low"])/voxel_struct["a_size"])
    elevation_capacity = np.floor((sensorcon["e_high"]-sensorcon["e_low"])/voxel_struct["e_size"])
    radius_capacity = np.floor((sensorcon["r_high"]-sensorcon["r_low"])/voxel_struct["r_size"])
    total_voxels = azimuth_capacity * elevation_capacity * radius_capacity
    out_ratio = vx.shape[0] / total_voxels
    
    return out_ratio
  
### Driver functions below ###
def data_rate_vista_automated(
    sensorcon_path: str, 
    vistaoutput_path: str, 
    prepad_output: bool = True, 
    enable_graphical: bool = True,
    enable_regression: bool = True,
    regression_power: int = 9,
    enable_resolution: bool = False,
    resolution: int = 1
    ) -> None:
    
    f = open(sensorcon_path)
    data = json.load(f)
    
    #Making sensor parameters so that we can find the accurate volume ratio.
    data["e_high"] += 90;
    data["e_low"] += 90;
    data["a_high"] += 180;
    data["a_low"] += 180;        

    sensor_range = data["r_high"] # Meters.
    padding_size = sensor_range # Meters.  

    # Set spherical voxel size from sensor precision:
    voxel_asize = data["horizAngRes"]
    voxel_esize = data["verticAngRes"]
    azimuth_fov = data["a_high"] - data["a_low"]
    elevation_fov = data["e_high"] - data["e_low"]

    # Refresh rate of the sensor. Needed for calculations.
    refresh_rate = 20  # In hertz
    
    # SNR is the signal to noise ratio. I don't know where to find these values.
    snrMax = 12  # I don't know the correct ballpark for this value.
    
    # Bit per measurements
    bitspermeasurements = 12 # Unit of bit.
    
    # just the r precision, which should be chosen by the user.
    voxel_rsize = 0.1    # in meters. Edit if you need to.
    
    point_density = 1
    
    ## Getting volume of the max sensor range
    # Doing so for data rate calculations
    max_volume = (1/3)*(math.pow(data["r_high"],3) - math.pow(data["r_low"],3))\
        *(np.cos(np.deg2rad(data["e_low"]))-np.cos(np.deg2rad(data["e_high"])))\
            *(np.deg2rad(data["a_high"])-np.deg2rad(data["a_low"]))
            
    file_regex = "output_*.txt"
    
    numScenes = len(vistaoutput_path)
    
    path = []
    total_scenes = []
    observers = []
    outmatrix_volume = []
    outmatrix_count = []
    
    for i in range(numScenes):
        path.append(os.path.join(vistaoutput_path[i], file_regex))
        
        total_scenes.append(len(glob.glob(path[-1])))
        
        #total_scenes is how many scenes there are in a folder
        #observers is how many scenes we intend to analyse
        if enable_resolution:
            #integer divison to round down so that you wont get extra observers that might crash the program
            observers.append(total_scenes[-1] // resolution)  
        else:
            observers.append(total_scenes[-1]) 

        outmatrix_volume.append(np.zeros([observers[-1], 2]))
        outmatrix_count.append(np.zeros([observers[-1], 2]))
   
    numCores = mp.cpu_count() - 1   
    
    for itr in range(numScenes):
        print('\nWorking on scene: ' + str(vistaoutput_path[itr]))
        # Read each of the scenes into memory in parallel, but can be configured to read once every few scenes
        #find smallest frame number in folder, program assumes smallest frame will be used in the graph analysis
        #need to find this so that program knows at which index to insert data into outmatrix
        smallest = math.inf
        for filename in os.listdir(vistaoutput_path[itr]):
            numFrame = get_first_number(filename)    
            if smallest > numFrame:
                smallest = numFrame
        upperbound = smallest + observers[itr] * resolution
                
        with mp.Pool(numCores) as p:
            inputData = [(voxel_rsize, voxel_asize, voxel_esize, data,i\
                        ,vistaoutput_path[itr],point_density,max_volume) for i in range(smallest, upperbound,resolution)]
            results = []
            with tqdm(total=len(inputData), desc="Processing Volume") as pbar:
                for result in p.imap(multiprocessed_vol_funct, inputData):
                    #result[0] is i in line 119. This subtraction is done so that result[1] can be inserted into
                    #outmatrix_volume at the proper index, which starts from 0, rather than at the variable "smallest"
                    outmatrix_volume[itr][(result[0] - smallest)//resolution] = result[1]
                    results.append(result)
                    pbar.update()
        
        with mp.Pool(numCores) as p:
            inputData = [(voxel_rsize, voxel_asize, voxel_esize, data,i\
                        ,vistaoutput_path[itr],point_density) for i in range(smallest, upperbound,resolution)]
            results = []
            with tqdm(total=len(inputData), desc="Processing Count") as pbar:
                for result in p.imap(multiprocessed_count_funct, inputData):
                    #result[0] is i in line 133. This subtraction is done so that result[1] can be inserted into
                    #outmatrix_count at the proper index, which starts from 0, rather than at the variable "smallest"
                    outmatrix_count[itr][(result[0] - smallest)//resolution] = result[1]
                    results.append(result)
                    pbar.update()

    print('\nDone!')
    
    ## Obtain delta/deltamax graph  
    outmatrix_volume2 = copy.deepcopy(outmatrix_volume)
    outmatrix_count2 = copy.deepcopy(outmatrix_count)

    for i in range(numScenes):
        outmatrix_volume2[i][:, 1] = outmatrix_volume2[i][:, 1] / np.max(outmatrix_volume2[i][:, 1])
        outmatrix_count2[i][:, 1] = outmatrix_count2[i][:, 1] / np.max(outmatrix_count2[i][:, 1])

    #calculating rolling average
    def rolling_average(input,col):
        window = 10
        average_y = []

        #for j in range(numScenes):
        for ind in range(len(input[:,col]) - window + 1):
            average_y.append(np.mean(input[ind:ind+window,col]))
        #print(len(average_y))
        
        for ind in range(window - 1):
            average_y.insert(0, np.nan)        
        
        return average_y
    
    outmatrix_volume_ave = []
    outmatrix_volume2_ave = []
    outmatrix_count_ave = []
    outmatrix_count2_ave = []
    
    for itr in range(numScenes):
        outmatrix_volume_ave.append(rolling_average(outmatrix_volume[itr],1))
        outmatrix_volume2_ave.append(rolling_average(outmatrix_volume2[itr],1))
        outmatrix_count_ave.append(rolling_average(outmatrix_count[itr],1))
        outmatrix_count2_ave.append(rolling_average(outmatrix_count2[itr],1))
    
    #complementary_colours = [['-r','-c'],['-g','-m'],['-b','-y']]
    complementary_colours = [['r','c'],['g','m'],['b','y']]
    
    ## Making graph
    if enable_graphical:
        if enable_regression:
            
            # Need to add main title and axis titles    
            fig1, ax1 = plt.subplots()
            fig1.canvas.manager.set_window_title('Volume Method') 
            fig1.suptitle("Data ratio of volumetric voxelization method", fontsize=12)
            ax1.set_ylabel(f"volume ratio (volume of occupied voxel/total volume in sensor) {get_folder(vistaoutput_path[0])}",color='r')
            ax1.tick_params(axis='y', colors='r')
            for i in range(numScenes):
                if i == 0:           
                    #ORIGINAL PLOT
                    #ax1.plot(outmatrix_volume[i][:, 0], outmatrix_volume[i][:, 1],\
                    #   f'r', label=f'Original)')
                    #ROLLING AVERAHE
                    ax1.plot(outmatrix_volume[i][:, 0], outmatrix_volume_ave[i],\
                        f'g', label=f'Rolling Average')
                    #BEST FIT LINE
                    poly, residual, _, _, _ = np.polyfit(outmatrix_volume[i][:, 0], outmatrix_volume[i][:, 1],\
                        deg=regression_power, full=True)
                    ax1.plot(outmatrix_volume[i][:, 0], np.polyval(poly, outmatrix_volume[i][:, 0]),\
                        f'b',\
                            label=f'Fitted: {get_folder(vistaoutput_path[i])}')
                else:
                    ax_new = ax1.twinx()
                    #ORIGINAL PLOT
                    #ax1.plot(outmatrix_volume[i][:, 0], outmatrix_volume[i][:, 1],\
                    #    f'r', label=f'Original)')
                    #ROLLING AVERAGE
                    ax_new.plot(outmatrix_volume[i][:, 0], outmatrix_volume_ave[i],\
                        f'g', label=f'Rolling Average')                    
                    #BEST FIT LINE
                    poly, residual, _, _, _ = np.polyfit(outmatrix_volume[i][:, 0], outmatrix_volume[i][:, 1],\
                        deg=regression_power, full=True)
                    ax_new.plot(outmatrix_volume[i][:, 0], np.polyval(poly, outmatrix_volume[i][:, 0]),\
                        f'b',\
                            label=f'Fitted: {get_folder(vistaoutput_path[i])}')
                    #Setting new Y-axis
                    ax_new.set_ylabel(f"volume ratio (volume of occupied voxel/total volume in sensor) {get_folder(vistaoutput_path[i])}"\
                        , color=complementary_colours[np.mod(i,3)][0])
                    ax_new.tick_params(axis='y', colors=complementary_colours[np.mod(i,3)][0])   
                    
                    offset = (i - 1) * 0.7
                    ax_new.spines['right'].set_position(('outward', offset * 100))                 
                
            plt.xlabel("distance (m)")
            fig1.legend()
            #plt.ylabel("volume ratio (volume of occupied voxel/total volume in sensor)")
            fig1.tight_layout()
            #plt.show() 
            
            fig2, ax2 = plt.subplots()
            fig2.canvas.manager.set_window_title('Simple Method') 
            fig2.suptitle("Data ratio of simple voxelization method", fontsize=12)
            ax2.set_ylabel(f"voxel count ratio (number of occupied voxel/total count in sensor) {get_folder(vistaoutput_path[0])}",color='r')
            ax2.tick_params(axis='y', colors='r')            
            for i in range(numScenes):
                if i == 0:
                    #ORIGINAL PLOT
                    ax2.plot(outmatrix_count[i][:, 0], outmatrix_count[i][:, 1],\
                        f'r', label=f'Original: {get_folder(vistaoutput_path[i])}')
                    #ROLLING AVERAGE
                    ax2.plot(outmatrix_count[i][:, 0], outmatrix_count_ave[i],\
                        f'g', label=f'Rolling Average')                    
                    #BEST FIT LINE
                    poly, residual, _, _, _ = np.polyfit(outmatrix_count[i][:, 0], outmatrix_count[i][:, 1],\
                        deg=regression_power, full=True)
                    ax2.plot(outmatrix_count[i][:, 0], np.polyval(poly, outmatrix_count[i][:, 0]),\
                        f'b',\
                            label=f'Fitted: {get_folder(vistaoutput_path[i])}')
                    
                else:
                    ax2_new = ax2.twinx()
                    #ORIGINAL PLOT
                    ax2_new.plot(outmatrix_count[i][:, 0], outmatrix_count[i][:, 1],\
                        f'r', label=f'Original: {get_folder(vistaoutput_path[i])}')
                    #ROLLING AVERAGE
                    ax2_new.plot(outmatrix_count[i][:, 0], outmatrix_count_ave[i],\
                        f'g', label=f'Rolling Average')                    
                    #BEST FIT LINE
                    poly, residual, _, _, _ = np.polyfit(outmatrix_count[i][:, 0], outmatrix_count[i][:, 1],\
                        deg=regression_power, full=True)
                    ax2_new.plot(outmatrix_count[i][:, 0], np.polyval(poly, outmatrix_count[i][:, 0]),\
                        f'b',\
                            label=f'Fitted:  {get_folder(vistaoutput_path[i])}')                    
                    #Setting new Y-axis
                    ax2_new.set_ylabel(f"voxel count ratio (number of occupied voxel/total count in sensor) {get_folder(vistaoutput_path[i])}"\
                        , color=complementary_colours[np.mod(i,3)][0])
                    ax2_new.tick_params(axis='y', colors=complementary_colours[np.mod(i,3)][0])   
                    
                    offset = (i - 1) * 0.7
                    ax2_new.spines['right'].set_position(('outward', offset * 100))  
                    
            plt.xlabel("distance (m)")
            #plt.ylabel("voxel count ratio (number of occupied voxel/total count in sensor)")
            fig2.legend()
            fig2.tight_layout()
            
            #plt.show(block=False)
            #plt.show()             
            
            fig3, ax3 = plt.subplots()
            fig3.canvas.manager.set_window_title('Volume Method') 
            fig3.suptitle("Delta ratio of volumetric voxelization method", fontsize=12)
            ax3.set_ylabel(f"delta ratio (delta/max delta) {get_folder(vistaoutput_path[0])}",color='r')
            ax3.tick_params(axis='y', colors='r')
            for i in range(numScenes):
                if i == 0:
                    #ORIGINAL PLOT
                    ax3.plot(outmatrix_volume2[i][:, 0], outmatrix_volume2[i][:, 1],\
                        f'r', label=f'Original: {get_folder(vistaoutput_path[i])}')
                    #ROLLING AVERAGE
                    ax3.plot(outmatrix_volume2[i][:, 0], outmatrix_volume2_ave[i],\
                        f'g', label=f'Rolling Average')                     
                    #BEST FIT LINE
                    poly, residual, _, _, _ = np.polyfit(outmatrix_volume2[i][:, 0], outmatrix_volume2[i][:, 1],\
                        deg=regression_power, full=True)
                    ax3.plot(outmatrix_volume2[i][:, 0], np.polyval(poly, outmatrix_volume2[i][:, 0]),\
                        f'b',\
                            label=f'Fitted:  {get_folder(vistaoutput_path[i])}')
                else:
                    ax3_new = ax3.twinx()
                    ax3_new.plot(outmatrix_volume2[i][:, 0], outmatrix_volume2[i][:, 1],\
                        f'r', label=f'Original: {get_folder(vistaoutput_path[i])}')
                    #ROLLING AVERAGE
                    ax3_new.plot(outmatrix_volume2[i][:, 0], outmatrix_volume2_ave[i],\
                        f'g', label=f'Rolling Average')                     
                    #BEST FIT LINE
                    poly, residual, _, _, _ = np.polyfit(outmatrix_volume2[i][:, 0], outmatrix_volume2[i][:, 1],\
                        deg=regression_power, full=True)
                    ax3_new.plot(outmatrix_volume2[i][:, 0], np.polyval(poly, outmatrix_volume2[i][:, 0]),\
                        f'b',\
                            label=f'Fitted:  {get_folder(vistaoutput_path[i])}')
                    #Setting new Y-axis
                    ax3_new.set_ylabel(f"delta ratio (delta/max delta) {get_folder(vistaoutput_path[i])}"\
                        , color=complementary_colours[np.mod(i,3)][0])
                    ax3_new.tick_params(axis='y', colors=complementary_colours[np.mod(i,3)][0])   
                    
                    offset = (i - 1) * 0.7
                    ax3_new.spines['right'].set_position(('outward', offset * 100))  
            
            plt.xlabel("distance (m)")
            #plt.ylabel("delta ratio (delta/max delta)")
            fig3.legend()
            fig3.tight_layout()
        
            #plt.show(block=False)
            #plt.show()
            
            fig6, ax6 = plt.subplots()
            fig6.canvas.manager.set_window_title("Simple method") 
            fig6.suptitle("Delta ratio of simple voxelization method", fontsize=12)
            ax6.set_ylabel(f"delta ratio (delta/max delta) {get_folder(vistaoutput_path[0])}",color='r')
            ax6.tick_params(axis='y', colors='r')
            for i in range(numScenes):
                if i == 0:
                    #ORIGINAL PLOT
                    ax6.plot(outmatrix_count2[i][:, 0], outmatrix_count2[i][:, 1],\
                        f'r', label=f'Original: {get_folder(vistaoutput_path[i])}')
                    #ROLLING AVERAGE
                    ax6.plot(outmatrix_count2[i][:, 0], outmatrix_count2_ave[i],\
                        f'g', label=f'Rolling Average')                     
                    #BEST FIT LINE
                    poly, residual, _, _, _ = np.polyfit(outmatrix_count2[i][:, 0], outmatrix_count2[i][:, 1],\
                        deg=regression_power, full=True)
                    ax6.plot(outmatrix_count2[i][:, 0], np.polyval(poly, outmatrix_count2[i][:, 0]),\
                        f'b',\
                            label=f'Fitted:  {get_folder(vistaoutput_path[i])}')
                else:
                    ax6_new = ax3.twinx()
                    ax6_new.plot(outmatrix_count2[i][:, 0], outmatrix_count2[i][:, 1],\
                        f'r', label=f'Original: {get_folder(vistaoutput_path[i])}')
                    #ROLLING AVERAGE
                    ax6_new.plot(outmatrix_count2[i][:, 0], outmatrix_count2_ave[i],\
                        f'g', label=f'Rolling Average')                     
                    #BEST FIT LINE
                    poly, residual, _, _, _ = np.polyfit(outmatrix_count2[i][:, 0], outmatrix_count2[i][:, 1],\
                        deg=regression_power, full=True)
                    ax6_new.plot(outmatrix_count2[i][:, 0], np.polyval(poly, outmatrix_count2[i][:, 0]),\
                        f'b',\
                            label=f'Fitted:  {get_folder(vistaoutput_path[i])}')
                    #Setting new Y-axis
                    ax6_new.set_ylabel(f"delta ratio (delta/max delta) {get_folder(vistaoutput_path[i])}"\
                        , color=complementary_colours[np.mod(i,3)][0])
                    ax6_new.tick_params(axis='y', colors=complementary_colours[np.mod(i,3)][0])   
                    
                    offset = (i - 1) * 0.7
                    ax6_new.spines['right'].set_position(('outward', offset * 100))  
            
            plt.xlabel("distance (m)")
            #plt.ylabel("delta ratio (delta/max delta)")
            fig6.legend()
            fig6.tight_layout() 
        else:
            #TO UPDATE
            '''
            fig1 = plt.figure("Volume method (data ratio)")
            fig1.suptitle("Data ratio of volumetric voxelization method", fontsize=12)
            for i in range(numScenes):
                plt.plot(outmatrix_volume[i][:, 0], outmatrix_volume[i][:, 1], f'{complementary_colours[np.mod(i,3)][0]}')
            plt.xlabel("distance (m)")
            plt.ylabel("volume ratio (volume of occupied voxel/total volume in sensor)")

            #plt.show(block=False)
            plt.show() 
            
            fig2 = plt.figure("Simple method")
            fig2.suptitle("Data ratio of simple voxelization method", fontsize=12)
            for i in range(numScenes):
                plt.plot(outmatrix_count[i][:, 0], outmatrix_count[i][:, 1], f'{complementary_colours[np.mod(i,3)][0]}')
            plt.xlabel("distance (m)")
            plt.ylabel("voxel count ratio (number of occupied voxel/total count in sensor)")
            
            #plt.show(block=False)
            #plt.show() 
            
            fig3 = plt.figure("Volume method (delta ratio)")
            fig3.suptitle("Delta ratio of volumetric voxelization method", fontsize=12)
            for i in range(numScenes):
                plt.plot(outmatrix_volume2[i][:, 0], outmatrix_volume2[i][:, 1], f'{complementary_colours[np.mod(i,3)][0]}')
            plt.xlabel("distance (m)")
            plt.ylabel("delta ratio (delta/max delta)")
        
            #plt.show(block=False) 
            #plt.show()            
            '''
            
    ## Data rate calculations
    # The calculation is the same one present on the paper we are basing this
    # program off of. Look there to find out how it works.

    # Done in steps to avoid matlab errors.
    # For outmatrix 1
    datarate_buffer = (32*sensor_range*azimuth_fov*elevation_fov*refresh_rate*bitspermeasurements)\
        /(3*voxel_asize*voxel_esize*voxel_rsize*snrMax)
    
    an_data_rate = []
    an_data_rate2 = []
    
    for i in range(numScenes):
        log_inverse_delta = np.log((1/(2*outmatrix_volume[i][:, 1])))
        delta_log_inverse_delta = outmatrix_volume[i][:, 1] * log_inverse_delta
        an_data_rate.append(np.transpose(np.array([delta_log_inverse_delta * datarate_buffer])))

        log_inverse_delta2 = np.log((1/(2*outmatrix_count[i][:, 1])))
        delta_log_inverse_delta2 = outmatrix_count[i][:, 1] * log_inverse_delta2
        an_data_rate2.append(np.transpose(np.array([delta_log_inverse_delta2 * datarate_buffer])))

    an_data_rate_ave = []
    an_data_rate2_ave = []
    
    for itr in range(numScenes):
        an_data_rate_ave.append(rolling_average(an_data_rate[itr],0))
        an_data_rate2_ave.append(rolling_average(an_data_rate2[itr],0))

    ## Datarate graphs
    if enable_graphical:
        if enable_regression:
            # Need to add main title and axis titles     
            fig4, ax4 = plt.subplots()
            fig4.canvas.manager.set_window_title('Volume method datarate') 
            fig4.suptitle("Data rate of volumetric voxelization method", fontsize=12)
            ax4.set_ylabel(f"Atomic norm Data rate {get_folder(vistaoutput_path[0])}",color='r')
            ax4.tick_params(axis='y', colors='r')
            for i in range(numScenes):
                if i == 0:
                    #ORIGINAL PLOT
                    ax4.plot(outmatrix_volume[i][:, 0], an_data_rate[i][:, 0],\
                        f'r', label=f'Original: {get_folder(vistaoutput_path[i])}')
                    #ROLLING AVERAGE
                    ax4.plot(outmatrix_volume[i][:, 0], an_data_rate_ave[i],\
                        f'g', label=f'Rolling Average')                      
                    #BEST FIT LINE
                    poly, residual, _, _, _ = np.polyfit(outmatrix_volume[i][:, 0], an_data_rate[i][:, 0], deg=regression_power, full=True)
                    ax4.plot(outmatrix_volume[i][:, 0], np.polyval(poly, outmatrix_volume[i][:, 0]),\
                        f'b',\
                            label=f'Fitted: {get_folder(vistaoutput_path[i])}')
                else:
                    ax4_new = ax4.twinx()
                    #ORIGINAL PLOT
                    ax4_new.plot(outmatrix_volume[i][:, 0], an_data_rate[i][:, 0],\
                        f'r', label=f'Original: {get_folder(vistaoutput_path[i])}')
                    #ROLLING AVERAGE
                    ax4_new.plot(outmatrix_volume[i][:, 0], an_data_rate_ave[i],\
                        f'g', label=f'Rolling Average')                      
                    #BEST FIT LINE
                    poly, residual, _, _, _ = np.polyfit(outmatrix_volume[i][:, 0], an_data_rate[i][:, 0], deg=regression_power, full=True)
                    ax4_new.plot(outmatrix_volume[i][:, 0], np.polyval(poly, outmatrix_volume[i][:, 0]),\
                        f'b',\
                            label=f'Fitted: {get_folder(vistaoutput_path[i])}')
                    #Setting new Y-axis
                    ax4_new.set_ylabel(f"Atomic norm Data rate {get_folder(vistaoutput_path[i])}"\
                        , color=complementary_colours[np.mod(i,3)][0])
                    ax4_new.tick_params(axis='y', colors=complementary_colours[np.mod(i,3)][0])   
                    
                    offset = (i - 1) * 0.7
                    ax4_new.spines['right'].set_position(('outward', offset * 100))
                     
            plt.xlabel("distance (m)")
            #plt.ylabel("Atomic norm Data rate")
            fig4.legend()
            fig4.tight_layout()
            
            #plt.show(block=False)
            #plt.show()
            
            fig5, ax5 = plt.subplots()
            fig5.canvas.manager.set_window_title("Simple method datarate") 
            fig5.suptitle("Data rate of simple voxelization method", fontsize=12)
            ax5.set_ylabel(f"Atomic norm Data rate {get_folder(vistaoutput_path[0])}",color='r')
            ax5.tick_params(axis='y', colors='r')
            for i in range(numScenes):
                if i == 0:
                    #ORIGINAL PLOT
                    ax5.plot(outmatrix_count[i][:, 0], an_data_rate2[i][:, 0],\
                        f'r', label=f'Original: {get_folder(vistaoutput_path[i])}')
                    #ROLLING AVERAGE
                    ax5.plot(outmatrix_count[i][:, 0], an_data_rate2_ave[i],\
                        f'g', label=f'Rolling Average')                      
                    #BEST FIT LINE
                    poly, residual, _, _, _ = np.polyfit(outmatrix_count[i][:, 0], an_data_rate2[i][:, 0], deg=regression_power, full=True)
                    ax5.plot(outmatrix_count[i][:, 0], np.polyval(poly, outmatrix_count[i][:, 0]),\
                        f'b',\
                            label=f'Fitted: {get_folder(vistaoutput_path[i])}')
                else:
                    ax5_new = ax5.twinx()
                    #ORIGINAL PLOT
                    ax5_new.plot(outmatrix_count[i][:, 0], an_data_rate2[i][:, 0],\
                        f'r', label=f'Original: {get_folder(vistaoutput_path[i])}')
                    #ROLLING AVERAGE
                    ax5_new.plot(outmatrix_count[i][:, 0], an_data_rate2_ave[i],\
                        f'g', label=f'Rolling Average')                      
                    #BEST FIT LINE
                    poly, residual, _, _, _ = np.polyfit(outmatrix_count[i][:, 0], an_data_rate2[i][:, 0], deg=regression_power, full=True)
                    ax5_new.plot(outmatrix_count[i][:, 0], np.polyval(poly, outmatrix_count[i][:, 0]),\
                        f'b',\
                            label=f'Fitted: {get_folder(vistaoutput_path[i])}')
                    #Setting new Y-axis
                    ax5_new.set_ylabel(f"Atomic norm Data rate {get_folder(vistaoutput_path[i])}"\
                        , color=complementary_colours[np.mod(i,3)][0])
                    ax5_new.tick_params(axis='y', colors=complementary_colours[np.mod(i,3)][0])   
                    
                    offset = (i - 1) * 0.7
                    ax5_new.spines['right'].set_position(('outward', offset * 100))
                    
            plt.xlabel("distance (m)")
            #plt.ylabel("Atomic norm Data rate")   
            fig5.legend()
            fig5.tight_layout()    
            
            #plt.show(block=False)
            #plt.show() 
        else:
            #TO UPDATE
            '''
            fig4 = plt.figure("Volume method datarate")
            fig4.suptitle("Data rate of volumetric voxelization method", fontsize=12)
            for i in range(numScenes):
                plt.plot(outmatrix_volume[i][:, 0], an_data_rate[i][:, 0], f'{complementary_colours[np.mod(i,3)][0]}')
            plt.xlabel("distance (m)")
            plt.ylabel("Atomic norm Data rate")
            
            #plt.show(block=False)
            #plt.show()
            
            fig5 = plt.figure("Simple method datarate")
            fig5.suptitle("Data rate of simple voxelization method", fontsize=12)
            for i in range(numScenes):
                plt.plot(outmatrix_count[i][:, 0], an_data_rate2[i][:, 0], f'{complementary_colours[np.mod(i,3)][0]}')
            plt.xlabel("distance (m)")
            plt.ylabel("Atomic norm Data rate")        
            

            #plt.show(block=False)   
            #plt.show()    
            '''    
    plt.show()

def main():
    args = file_tools.parse_cmdline_args()
    sensorcon_path = file_tools.obtain_sensor_path(args)
    path2scenes = file_tools.obtain_multiple_scene_path(args)
    
    data_rate_vista_automated(
        sensorcon_path=sensorcon_path,
        vistaoutput_path=path2scenes, 
        prepad_output=True, 
        enable_graphical=True,
        enable_regression=True,
        regression_power=10
        )
    return

if __name__ == "__main__":
    main()
