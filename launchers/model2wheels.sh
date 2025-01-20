#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun model2wheels model2wheels.py

# wait for app to end
dt-launchfile-join
