#!/bin/bash

ulimit -n 65536
source $(dirname "${CONDA_PYTHON_EXE}")/activate male
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=$1
shift

python \
main_finetune.py \
--config_file configs/finetune/Market-1501/mae_inet_lup_vitb_ep800_ratio_optimized/baseline_384.yaml \
$@