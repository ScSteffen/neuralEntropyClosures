docker run  --gpus all --cpuset-cpus=0-4 -i --rm -t -v $(pwd)/..:/mnt scsteffen/neural_entropy:latest /bin/bash
