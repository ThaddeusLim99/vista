# Documentation for the Vista simulator

## Overview

The Vista simulator takes an input point cloud, and simulates an autonomous vehicle as it 'drives' through the point cloud itself, from a trajectory that is manually determined from the input point cloud itself (see [here]() for more info on the trajectory). This is a fork of the Vista simulator with features added on to calculate the data rate of a simulated AV.

- The trajectory details the orientation, and position of the vehicle at a discrete interval of road points on the road itself.
  - We take 1 meter per point along the road; if a point cloud has 4000 points then it is 4km long.

At a road point, we will segment the entire point cloud down to the sensor FOV:
![Sensor FOV at 450m](/images/velodyne_alpha_128_at450m.png "The point cloud (in grey) within the FOV range (points in white) is then segmented")
, perform a rigid body transformation on the point cloud itself to local coordinates (using the trajectory), and then simulate the LiDAR sensor on this transformed & segmented point cloud with the Vista simulator.
