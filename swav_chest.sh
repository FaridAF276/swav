#! /bin/bash
##Download dataset

wget https://data.mendeley.com/public-files/datasets/jctsfj2sfn/files/148dd4e7-636b-404b-8a3c-6938158bc2c0/file_downloaded && \
unzip file_downloaded
mkdir swav_checkpoint
splitfolders --output ChestX --ratio .8 .1 .1 --move \
-- COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset
time python dataset_prep.py \
--dataset_dir ChestX \
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

zip -r chest_swav_pretext.zip swav_checkpoint
./gdrive upload chest_swav_pretext.zip
mkdir swav_ssl_checkpoint
time python -m torch.distributed.launch --nproc_per_node=8 eval_semisup.py \
--data_path downstream \
--pretrained swav_checkpoint/swav_2ep_pretrain.pth.tar \
--epochs 1 \
--labels_perc "10" \
--lr 0.01 \
--lr_last_layer 0.2\
--dump_path swav_ssl_checkpoint
zip -r chest_swav_downstr.zip swav_ssl_checkpoint
./gdrive upload chest_swav_downstr.zip
