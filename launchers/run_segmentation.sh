#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch nodes in parallel
rosrun image_processor image_processor_node.py &
rosrun seg2wheels seg2wheels.py &

# wait for app to end
dt-launchfile-join
