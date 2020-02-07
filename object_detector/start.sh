#!/bin/bash
source /opt/intel/openvino_2019.3.376/bin/setupvars.sh 3.5
export PYTHONPATH=$PYTHONPATH:/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

echo Your container args are: "$@"
python3 object_detector.py "$@"
