#!/bin/bash

set -e

SCRIPT_DIR=$(dirname "$0")

if [ -z "$1" ]; then
    echo "Usage: ./run.sh <target>"
    echo "  mnist-tf          tf/mnist/mnist-tf.py"
    echo "  salary-tf         tf/salary/salary-tf.py"
    echo "  salary-torch      torch/salary/salary-torch.py"
    echo "  medical-torch     torch/medical-insurance/medical-insurance-torch.py"
    echo "  bestsellers       torch/bestsellers/model.py"
    exit 1
elif [ "$1" == "mnist-tf" ]; then
    TARGET_FILE=tf/mnist/mnist-tf.py
elif [ "$1" == "salary-tf" ]; then
    TARGET_FILE=tf/salary/salary-tf.py
elif [ "$1" == "salary-torch" ]; then
    TARGET_FILE=torch/salary/salary-torch.py
elif [ "$1" == "medical-torch" ]; then
    TARGET_FILE=torch/medical-insurance/medical-insurance-torch.py
elif [ "$1" == "bestsellers-torch" ]; then
    TARGET_FILE=torch/bestsellers/model.py
else
    echo "invalid target: $1"
    exit 1
fi

cd "$SCRIPT_DIR/$(dirname $TARGET_FILE)"
python3 $(basename $TARGET_FILE)
