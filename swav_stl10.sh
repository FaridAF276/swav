#!/bin/bash
# cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key; \
# wget -nc https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast; \
# ./vast start instance ${VAST_CONTAINERLABEL:2} && \
# bash -e swav/swav_stl10.sh && \
# ./vast stop instance ${VAST_CONTAINERLABEL:2}
##Download dataset imagefolder
cd swav
gdown --fuzzy https://drive.google.com/file/d/1B0GLPjsXgtWLhV5SvBT1Kti1QHKWql2t/view?usp=sharing
unzip -n stl10.zip -d stl10
rm -rf stl10/unlabelled
mkdir -p swav_checkpoint

time python dataset_prep.py \
--dataset_dir stl10 \
--percentage 0.2

time python -m torch.distributed.launch --nproc_per_node=4 main_swav.py \
--data_path pretext/train \
--epochs 500 \
--base_lr 0.01 \
--final_lr 0.0006 \
--warmup_epochs 0 \
--batch_size 32 \
--epsilon 0.03 \
--dump_path swav_checkpoint \
--size_crops 32 64 \
--nmb_crops 2 3 \
--min_scale_crops 0.14 0.05 \
--nmb_prototypes 10 \
--max_scale_crops 1. 0.14 \
--use_fp16 true \
--freeze_prototypes_niters 5005 \
--queue_length 3840 \
--temperature 0.5 \
--epoch_queue_starts 15
mkdir -p swav_ssl_checkpoint
time python -m torch.distributed.launch --nproc_per_node=4 eval_semisup.py \
--data_path downstream \
--pretrained swav_checkpoint/checkpoint.pth.tar \
--epochs 1 \
--labels_perc "10" \
--lr 0.06 \
--lr_last_layer 0.2 \
--dump_path swav_ssl_checkpoint
zip -r stl10_swav_pretext.zip swav_checkpoint
zip -r stl10_swav_downstr.zip swav_ssl_checkpoint
cd
./gdrive upload stl10_swav_pretext.zip
./gdrive upload stl10_swav_downstr.zip
