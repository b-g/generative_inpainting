#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd "$SCRIPTPATH"
for f in output/Fast-and-Furious-2019/frames-512/*.jpg; do
  filename=$(basename "$f" .jpg)
  output="output/Fast-and-Furious-2019/deepfill-512/${filename}.png"
  if [ ! -f "$output" ]; then
    echo $filename
    python3 test.py \
      --image "$f" \
      --mask "output/Fast-and-Furious-2019/masked-512/${filename}.png" \
      --output "$output" \
      --checkpoint_dir model_logs/release_places2_256 \
      /
  fi
done
