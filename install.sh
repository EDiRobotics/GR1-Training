#!/bin/bash
pip install -r requirements.txt
pip install transformers==4.11.0

sudo apt-get -y install libegl1-mesa libegl1
sudo apt-get -y install libgl1
sudo apt-get -y install libosmesa6-dev
sudo apt-get -y install patchelf