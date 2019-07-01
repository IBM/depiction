#!/bin/sh

echo "Start jupyter-notebook on port 8888"

jupyter notebook --ip="0.0.0.0" --port=8888 \
    --no-browser --allow-root \
    --NotebookApp.token="${JUPYTER_TOKEN}" --notebook-dir="/workspace"
