#!/bin/bash

##Download dataset imagefolder
cd swav
gdown --fuzzy https://drive.google.com/file/d/1B0GLPjsXgtWLhV5SvBT1Kti1QHKWql2t/view?usp=sharing
unzip -n stl10.zip -d stl10
rm -rf stl10/unlabelled
mkdir -p swav_checkpoint

time python dataset_prep.py \
--dataset_dir stl10 \
--percentage 0.2

time python -m torch.distributed.launch --nproc_per_node=2 main_swav.py \
--data_path pretext/train \
--epochs 2 \
--base_lr 0.6 \
--final_lr 0.0006 \
--warmup_epochs 0 \
--batch_size 32 \
--dump_path swav_checkpoint \
--size_crops 224 96 \
--nmb_crops 2 6 \
--min_scale_crops 0.14 0.05 \
--nmb_prototypes 150 \
--max_scale_crops 1. 0.14 \
--use_fp16 true \
--freeze_prototypes_niters 5005 \
--queue_length 3840 \
--epoch_queue_starts 15

zip -r stl10_swav_pretext.zip swav_checkpoint
.~/gdrive upload stl10_swav_pretext.zip
mkdir -p swav_ssl_checkpoint
time python -m torch.distributed.launch --nproc_per_node=2 eval_semisup.py \
--data_path downstream \
--pretrained swav_checkpoint/checkpoint.pth.tar \
--epochs 1 \
--labels_perc "10" \
--lr 0.01 \
--lr_last_layer 0.2 \
--dump_path swav_ssl_checkpoint
zip -r stl10_swav_downstr.zip swav_ssl_checkpoint
.~/gdrive upload stl10_swav_downstr.zip
