# apt-get install -y git && git clone https://github.com/FaridAF276/swav.git && chmod +x swav/swav_imagenet.sh && bash -e swav/swav_imagenet.sh
#Setup everything to install apex correctly (configure vast.ai with cuda 10.1 and pytorch 1.4.0)
wget https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz
tar -xvf gdrive_2.1.1_linux_386.tar.gz
./gdrive about
apt-get install -y unzip zip git wget
apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash ~/Anaconda3-2022.05-Linux-x86_64.sh -b -p
cd swav
source ~/anaconda3/etc/profile.d/conda.sh # Or path to where your conda is
conda create -y --name=swav python=3.6.6
conda activate swav
conda update -y -n base -c defaults conda
conda install -y pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
# conda install -y -c conda-forge cudatoolkit-dev=10.1.243
conda install -y -c conda-forge opencv
conda install -y -c anaconda pandas=0.25.0
which pip
git clone "https://github.com/NVIDIA/apex"
cd apex
git checkout 4a1aa97e31ca87514e17c3cd3bbc03f4204579d0
python setup.py install --cuda_ext
python -c 'import apex; from apex.parallel import LARC' # should run and return nothing
python -c 'import apex; from apex.parallel import SyncBatchNorm; print(SyncBatchNorm.__module__)' # should run and return apex.parallel.optimized_sync_batchnorm 
cd ~/swav/
conda install -y -c conda-forge gdown
# tail -n +29 swav_imagenet.sh | bash

##Download dataset

gdown --fuzzy https://drive.google.com/file/d/1ny6vBH54X0qV07EsNhgddHFGYgoROswy/view?usp=sharing
unzip cifar10.zip
mkdir swav_checkpoint

time python dataset_preparation.py \
--dataset_dir imagenet \
--percentage 0.2

time python -m torch.distributed.launch --nproc_per_node=8 main_swav.py \
--data_path imagenet/train \
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
--data_path imagenet \
--pretrained swav_checkpoint/swav_2ep_pretrain.pth.tar \
--epochs 1 \
--labels_perc "10" \
--lr 0.01 \
--lr_last_layer 0.2\
--dump_path swav_ssl_checkpoint
zip -r imagenet_swav_downstr.zip swav_ssl_checkpoint
./gdrive upload imagenet_swav_downstr.zip
