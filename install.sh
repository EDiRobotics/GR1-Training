#!/bin/bash

pip3 install packaging==21.3
pip3 install setuptools==57.5.0
pip3 install flamingo_pytorch
pip3 install tensorboard
pip3 install ftfy
pip3 install regex
pip3 install tqdm
pip3 install matplotlib
pip3 install torch torchvision torchaudio
pip3 install transformers==4.11.0
pip3 install einops 
pip3 install lmdb 
pip3 install opencv-python
pip3 install av
pip3 install git+https://github.com/openai/CLIP.git

sudo apt-get -y install libegl1-mesa libegl1
sudo apt-get -y install libgl1
sudo apt-get -y install libosmesa6-dev
sudo apt-get -y install patchelf