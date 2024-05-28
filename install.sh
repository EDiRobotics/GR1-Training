#!/bin/bash
pip install -r requirements.txt

apt-get -y install libegl1-mesa libegl1
apt-get -y install libgl1
apt-get -y install libosmesa6-dev
apt-get -y install patchelf
