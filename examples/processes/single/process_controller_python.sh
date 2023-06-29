#!/bin/bash

### USER INPUT HERE ###
processes=6
LASFILE="VL-Veg-81604S_C1L1_08000_04000_y_trimmed.las"
JSONFILE=velodyne_alpha_128_front.json # Do not put quotes here
observer_height=1.8
PAD_OUTPUTS=true
RUN_OCCLUSION=true

#TODO Parse the sensor config file from shell script and then pass the variables to each process
# You can input the sensor .json for now
RESOLUTION=0.11
PITCH_MIN=-25
PITCH_MAX=15
YAW_MIN=-90
YAW_MAX=90
RANGE=245
CULLING_R=2

# Comment this out if you want to generate a trajectory or if you already have a pregenerated trajectory
python gen_traj.py --input examples/vista_traces/${LASFILE} --observer_height ${observer_height}

# Here you can run the vista output from a certain range if you need to
if $PAD_OUTPUTS
then
    # Run the entire road section, padded with the sensor range
    STARTFRAME=$RANGE
    #STARTFRAME=1145
    ENDFRAME=$((`cat examples/Trajectory/${LASFILE%.*}/forwards.csv | wc -l`-$RANGE))
else
    STARTFRAME=0
    ENDFRAME=`cat examples/Trajectory/${LASFILE%.*}/forwards.csv | wc -l`
    # ENDFRAME=6 # Uncomment this if you want to run in a certain range
fi

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
export CULLING_R

export LASFILE
export STARTFRAME
export ENDFRAME
export PAD_OUTPUTS
export RUN_OCCLUSION

start=1
end=$processes

echo "Computing output from road point ${STARTFRAME} to road point ${ENDFRAME}..."
echo "Input road section: ${LASFILE}"
echo "Input sensor configuration: ${JSONFILE}"
if $RUN_OCCLUSION
then
    echo "Running with occluded outputs!"
else
    echo "Running with unoccluded outputs!"
fi
echo
echo "Starting ${processes} processes..."
start_time=`date +%s`


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
if $RUN_OCCLUSION
then
    OUTPATH="examples/vista_traces/lidar_output/${LASFILE%.*}_resolution=${RESOLUTION}/"
else
    OUTPATH="examples/vista_traces/lidar_output/${LASFILE%.*}_unoccluded/"
fi

end_time=`date +%s`
echo "Vista simulation took $((${end_time}-${start_time})) seconds."

# Check if all Vista scenes within our range were generated
total_outputs=`ls "examples/vista_traces/lidar_output/${LASFILE%.*}_resolution=${RESOLUTION}" | wc -l`
expected_outputs=$(( $ENDFRAME - $STARTFRAME ))
if [[ $total_outputs -ne $expected_outputs ]]
then
    echo "Expected ${expected_outputs} outputs! (got ${total_outputs})"
    # exit 1
    # Find a way to generate the missing outputs
else
    echo "Generating graphs with Python..."
    python data_rate_vista.py --config ${SENSORPATH} --scenes ${OUTPATH}
fi

#FOR DEBUGGING MATLAB FUNCTION, USING ALREADY GENERATED VISTA OUTPUTS
# NOTE THAT YOU MAY HAVE TO EDIT THE OUTPUT FOLDER
# data_rate_vista_automated("/home/mohamed/Desktop/sensor-voxelization-cst/DataRate_fromCH/sensors/velodyne_alpha_128.json", "~/Desktop/vista/examples/vista_traces/lidar_output/03210N_C1R1_16000_20000_y_trimmed_resolution=0.11", true, true)
