#! /bin/bash
##Download dataset

gdown --fuzzy https://drive.google.com/file/d/1_dRbJEpMH7436l8aU4xrGHcFIE9i5TX7/view?usp=sharing
unzip tiny_imagenet_200.zip
mkdir swav_checkpoint

time python dataset_prep.py \
--dataset_dir imagenet \
--percentage 0.2

time python -m torch.distributed.launch --nproc_per_node=8 main_swav.py \
--data_path pretext/train \
--epochs 1 \
--base_lr 0.6 \
--final_lr 0.0006 \
--warmup_epochs 0 \
--batch_size 32 \
--dump_path swav_checkpoint \
--size_crops 224 96 \
--nmb_crops 2 6 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--use_fp16 true \
--freeze_prototypes_niters 5005 \
--queue_length 3840 \
--epoch_queue_starts 15

zip -r imagenet_swav_pretext.zip swav_checkpoint
./gdrive upload imagenet_swav_pretext.zip
mkdir swav_ssl_checkpoint
time python -m torch.distributed.launch --nproc_per_node=8 eval_semisup.py \
--data_path downstream \
--pretrained swav_checkpoint/swav_2ep_pretrain.pth.tar \
--epochs 1 \
--labels_perc "10" \
--lr 0.01 \
--lr_last_layer 0.2 \
--dump_path swav_ssl_checkpoint
zip -r imagenet_swav_downstr.zip swav_ssl_checkpoint
./gdrive upload imagenet_swav_downstr.zip
