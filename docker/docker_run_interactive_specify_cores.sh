docker run  --gpus all --cpuset-cpus=29-31 -i --rm -t -v $(pwd)/..:/mnt scsteffen/neural_entropy:latest /bin/bash
