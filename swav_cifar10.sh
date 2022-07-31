#!/bin/bash
#lr: 0.6
##Download dataset
cd swav
gdown --fuzzy https://drive.google.com/file/d/1ny6vBH54X0qV07EsNhgddHFGYgoROswy/view?usp=sharing
unzip -n cifar10.zip
mkdir -p swav_checkpoint && \
mkdir -p swav_ssl_checkpoint

time python dataset_prep.py \
--dataset_dir cifar10 \
--percentage 0.2

time python -m torch.distributed.launch --nproc_per_node=4 main_swav.py \
--data_path pretext/train \
--epochs 500 \
--base_lr 0.04 \
--final_lr 0.0006 \
--warmup_epochs 0 \
--batch_size 32 \
--dump_path swav_checkpoint \
--size_crops 32 \
--nmb_crops 2 \
--epsilon 0.03 \
--min_scale_crops 0.14 \
--max_scale_crops 1. \
--use_fp16 true \
--nmb_prototypes 30 \
--freeze_prototypes_niters 5005 \
--queue_length 3840 \
--temperature 0.5 \
--epoch_queue_starts 15

zip -r cifar10_swav_pretext.zip swav_checkpoint
.~/gdrive upload cifar10_swav_pretext.zip
mkdir -p swav_ssl_checkpoint
time python -m torch.distributed.launch --nproc_per_node=8 eval_semisup.py \
--data_path downstream \
--pretrained swav_checkpoint/checkpoint.pth.tar \
--epochs 1 \
--labels_perc "10" \
--lr 0.06 \
--lr_last_layer 0.2 \
--dump_path swav_ssl_checkpoint
zip -r cifar10_swav_downstr.zip swav_ssl_checkpoint
.~/gdrive upload cifar10_swav_downstr.zip