#! /bin/bash
##Download dataset
# cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key; \
# wget -nc https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast; \
# ./vast start instance ${VAST_CONTAINERLABEL:2} && \
# bash -e swav/swav_chest.sh && \
# ./vast stop instance ${VAST_CONTAINERLABEL:2}
cd swav
wget -nc https://data.mendeley.com/public-files/datasets/jctsfj2sfn/files/148dd4e7-636b-404b-8a3c-6938158bc2c0/file_downloaded && \
unzip -n file_downloaded
mkdir -p swav_checkpoint
splitfolders --output ChestX --ratio .8 .1 .1 --move \
-- COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset
time python dataset_prep.py \
--dataset_dir ChestX \
--percentage 0.2

time python -m torch.distributed.launch --nproc_per_node=2 main_swav.py \
--data_path pretext/train \
--epochs 500 \
--base_lr 0.01 \
--final_lr 0.0006 \
--warmup_epochs 0 \
--batch_size 8 \
--dump_path swav_checkpoint \
--size_crops 224 500 \
--nmb_crops 2 6 \
--epsilon 0.03 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--use_fp16 true \
--nmb_prototypes 6 \
--freeze_prototypes_niters 400 \
--queue_length 3840 \
--temperature 0.5 \
--epoch_queue_starts 30
mkdir -p swav_ssl_checkpoint
time python -m torch.distributed.launch --nproc_per_node=2 eval_semisup.py \
--data_path downstream \
--pretrained swav_checkpoint/checkpoint.pth.tar \
--epochs 200 \
--labels_perc "10" \
--lr 0.01 \
--lr_last_layer 0.2 \
--dump_path swav_ssl_checkpoint
zip -r chest_swav_pretext.zip swav_checkpoint
zip -r chest_swav_downstr.zip swav_ssl_checkpoint
cd 
./gdrive upload chest_swav_pretext.zip
./gdrive upload chest_swav_downstr.zip
