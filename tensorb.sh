#!/bin/bash 

echo "staring Tensorboard"

# echo "Activating conda environment ....!"
# source  activate idp3

# echo "Activating conda environment done!"
# conda info --env


echo "Tensorboard  started...!"


echo "Tensorboard  Folder:"$1
echo "Tensorboard  Port:"$2

tensorboard --logdir=$1  --port=$2



echo "Finished Tensorboard"
