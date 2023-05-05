#!/bin/bash

### USER INPUT HERE ###
processes=6
LASFILE="01617W_L1L2_L1L3_08000_04000.las"
JSONFILE=velodyne_alpha_128.json # Do not put quotes here
observer_height=1.8

#TODO Parse the sensor config file from shell script and then pass the variables to each process
# You can input the sensor .json for now
RESOLUTION=0.11
PITCH_MIN=-25
PITCH_MAX=15
YAW_MIN=-180
YAW_MAX=180
RANGE=245

# Comment this out if you want to generate a trajectory or if you already have a pregenerated trajectory
python gen_traj.py --input examples/vista_traces/${LASFILE} --observer_height ${observer_height}

# Here you can run the vista output from a certain range if you need to
STARTFRAME=0
ENDFRAME=`cat examples/Trajectory/${LASFILE%.*}/forwards.csv | wc -l`
ENDFRAME=6 # Uncomment this if you want to run in a certain range

### USER INPUT ENDS HERE ###

if (( $STARTFRAME > $ENDFRAME ))
then
    echo "Cannot have starting frame greater than end frame! (${STARTFRAME} > ${ENDFRAME})"
    exit 1
fi

export RESOLUTION
export PITCH_MIN
export PITCH_MAX
export YAW_MIN
export YAW_MAX
export RANGE

export LASFILE
export STARTFRAME
export ENDFRAME

start=1
end=$processes

echo "Computing output from road point ${STARTFRAME} to road point ${ENDFRAME}..."
echo "Input road section: ${LASFILE}"
echo "Input sensor configuration: ${JSONFILE}"
echo
echo "Starting ${processes} processes..."


for (( i=$start; i<=$processes; i++ ))
do
    #xterm -T "Process ${i}" -hold -e bash examples/processes/single/process_${i}.sh &
    xterm -T "Process ${i}" -e bash examples/processes/single/process_${i}.sh &

    echo "Process ${i} of ${end} started"
done

wait # Waits until every single process has finished

echo "All processes have finished."
echo


SENSORPATH=~/Desktop/sensor-voxelization-cst/DataRate_fromCH/sensors/${JSONFILE}
VISTAOUTPATH="~/Desktop/vista/examples/vista_traces/lidar_output/${LASFILE%.*}_resolution=${RESOLUTION}"

echo "Opening MATLAB..."
matlab -sd "~/Desktop/sensor-voxelization-cst/DataRate_fromCH" -r "data_rate_vista_automated('${SENSORPATH}', '${VISTAOUTPATH}')"

#FOR DEBUGGING MATLAB FUNCTION, USING ALREADY GENERATED VISTA OUTPUTS
# NOTE THAT YOU MAY HAVE TO EDIT THE OUTPUT FOLDER
# data_rate_vista_automated("/home/mohamed/Desktop/sensor-voxelization-cst/DataRate_fromCH/sensors/velodyne_alpha_128.json", "~/Desktop/vista/examples/vista_traces/lidar_output/01617W_L1L2_L1L3_08000_04000_resolution=0.11")