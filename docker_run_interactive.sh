docker run  --gpus all -i -t --rm --cpuset-cpus=0-3 -v $(pwd):/mnt tensorflow/tensorflow:devel-gpu /bin/bash
