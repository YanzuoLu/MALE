#!/bin/bash

ulimit -n 65536
source $(dirname "${CONDA_PYTHON_EXE}")/activate male
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=$1
shift

torchrun \
--rdzv_backend=c10d \
--rdzv_endpoint=localhost:29400 \
--nnodes=1 \
--nproc_per_node=4 \
main_pretrain.py \
--config_file configs/pretrain/mae/mae_inet_lup_vitb_ep400_ratio_optimized.yaml \
$@