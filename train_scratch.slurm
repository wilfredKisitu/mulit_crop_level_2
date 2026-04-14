echo "=== Job started on $(hostname) at $(data) ===="

source ~/.bashrc
conda activate dev-multicrop

echo "Conda environment $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

cd /home/nalwangar/wilfred/multi_crop_level_2

python -m model_train.train

echo "=== Jon finised at $(data) ==="