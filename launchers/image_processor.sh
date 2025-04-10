#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun image_processor image_processor_node.py

# wait for app to end
dt-launchfile-join
