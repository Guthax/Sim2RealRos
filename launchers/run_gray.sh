#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch nodes in parallel
rosrun gray2wheels gray2wheels.py

# wait for app to end
dt-launchfile-join
