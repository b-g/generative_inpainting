## Install
1. `nvidia-docker build -t generative-inpainting:v0 .`
2. Download the pretrained models e.g. Places2 and copy it to `model_logs`

## Running
./run-docker.sh

inside docker:

python3 test.py --image output/input.png --mask output/mask.png --output output/output.png --checkpoint_dir model_logs/release_places2_256

## Mask Image
value 255/255/255 indicates the mask, alpha is ignored