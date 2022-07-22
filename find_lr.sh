time python -m torch.distributed.launch --nproc_per_node=1 find_lr.py \
--data_path cifar10/train \
--epochs 2 \
--base_lr 0.6 \
--lr_min 0.4 \
--lr_max 0.6 \
--logspace \
--final_lr 0.0006 \
--warmup_epochs 0 \
--batch_size 32 \
--dump_path swav_checkpoint \
--size_crops 32 16 \
--nmb_crops 2 6 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--use_fp16 true \
--freeze_prototypes_niters 5005 \
--queue_length 3840 \
--epoch_queue_starts 15