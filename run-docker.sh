#!/bin/bash
DOCKER_VOLUMES+="-v $(pwd)/:/shared "
shift
docker run --runtime=nvidia -it $DOCKER_VOLUMES --workdir /shared generative-inpainting:v0 bash
