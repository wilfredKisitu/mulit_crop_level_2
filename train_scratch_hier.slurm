#!/bin/bash

echo "=== Job started on $(hostname) at $(date) ==="

source ~/.bashrc
conda activate dev-multicrop

echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

cd /home/nalwangar/wilfred/mulit_crop_level_2

python -m model_train.train_hier

echo "=== Job finished at $(date) ==="
