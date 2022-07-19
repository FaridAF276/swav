#! /bin/bash

# apt-get install -y git && git clone https://github.com/FaridAF276/swav.git && chmod +x swav/setup_env.sh && bash -e swav/setup_env.sh
#Setup everything to install apex correctly (configure vast.ai with cuda 10.1 and pytorch 1.4.0)
wget -nc https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz
tar -xvf gdrive_2.1.1_linux_386.tar.gz
./gdrive about
apt-get install -y unzip zip git wget
apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
wget -nc https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash ~/Anaconda3-2022.05-Linux-x86_64.sh -b -p
conda create -y --name=swav python=3.6.6 pandas=0.25.0 opencv
source ~/anaconda3/etc/profile.d/conda.sh # Or path to where your conda is
conda activate /opt/conda/envs/swav
conda install -y pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
# conda install -y -c conda-forge cudatoolkit-dev=10.1.243
which pip
cd swav
git clone "https://github.com/NVIDIA/apex"
cd apex
git checkout 4a1aa97e31ca87514e17c3cd3bbc03f4204579d0
python setup.py install --cuda_ext
python -c 'import apex; from apex.parallel import LARC' # should run and return nothing
python -c 'import apex; from apex.parallel import SyncBatchNorm; print(SyncBatchNorm.__module__)' # should run and return apex.parallel.optimized_sync_batchnorm 
cd ~/swav/
conda install -y -c conda-forge gdown