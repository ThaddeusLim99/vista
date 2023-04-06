#!/bin/bash

processes=1
LASFILE="01A02E_C1R1_16000_20000_START_HZ.las"

start=1
end=$processes


export LASFILE
for (( i=$start; i<=$processes; i++ ))
do
    xterm -T "Process ${i}" -hold -e bash examples/processes/single/process_${i}.sh &
    echo "Process ${i} of ${end} started"
done