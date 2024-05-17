# GR1-Training
Reproduced training script for GR-1: "Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation". It's current SOTA in [CALVIN Manipulation Benchmark](http://calvin.cs.uni-freiburg.de/) without using depth information.

The original implementation is [here](https://github.com/bytedance/GR-1) but there is no training script.

Star and cite this repo (and of course the original implementation) if you find it is helpful!

## Installation
Setup conda environment and install the CALVIN benchmark. 
```
source activate
conda create -n gr python=3.8
conda activate gr
git clone --recurse-submodules https://github.com/mees/calvin.git
pip install setuptools==57.5.0
cd calvin
cd calvin_env; git checkout main
cd ../calvin_models
sed -i 's/pytorch-lightning==1.8.6/pytorch-lightning/g' requirements.txt
sed -i 's/torch==1.13.1/torch/g' requirements.txt
cd ..
sh ./install.sh
cd ..```
```
Install this repository:
```
git clone https://github.com/EDiRobotics/GR1-Training
cd ./GR1-Training
sh ./install.sh
```

## Prepare Dataset
The [original CALVIN dataset](https://github.com/mees/calvin/tree/main/dataset) is too large (~500GB for each task) and contains trajectories data without any language annotation. Here, we only use the trajectories with annotations for training (the same as GR-1 and 3D Diffuser Actor). 

As an example, let's download the CALVIN debug dataset (1.3GB) and transfer it to our LMDB format.
```
wget http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip
unzip calvin_debug_dataset.zip
python calvin2lmdb.py --input_dir ./calvin_debug_dataset --output_dir ./calvin_lmdb
```
You can also download the processed LMDB dataset (ABC->D split) from Huggingface. The LMDB dataset only takes ~23GB, while the original ABC->D split takes 517GB. In this example, I use the tool of [HF-Mirror](https://hf-mirror.com/). You can set the environment variable `export HF_ENDPOINT=https://hf-mirror.com` to avoid the connection problem in some regions.
```
rm calvin_lmdb
apt install git-lfs aria2
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
./hfd.sh StarCycle/calvin_lmdb --dataset --tool aria2c -x 4
```
## Start Training & Evaluation
**The configuration parameters are saved in `config.json`**

You need to download the [weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) of the ViT visual encoder to the GR1-Training folder.

You can train from scratch or from a pretrained policy weight. The official weight is [here](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth), but directly finetuning this may need several epochs to recover the original performance (I try to mimic their training pipeline, but my implementation is still not the same). To use their weight, you need to slightly modify the checkpoint loading code in `Main.py`.

You can also use my finetuned weight, which is included in the [calvin_lmdb dataset](https://huggingface.co/datasets/StarCycle/calvin_lmdb) in HuggingFace.
```
mv ./calvin_lmdb/GR1_0.pth ./Save
```
Also set the CALVIN_ROOT environment variable:
```
export CALVIN_ROOT=<path to the calvin repo>
```
Now you can simply launch training with 
```
python Main.py
```
Before the training starts, it will first evaluate the policy. `Main.py` will also evaluate the policy for 1000 times after finishing 1 epoch of training. There is no seperated evaluation code.

## Detailed Analysis of the Network and API
Please refer to this [issue](https://github.com/bytedance/GR-1/issues/4)

## Multi-GPU Training & Evaluation
The code here is simplified and easy to adapt to any training pipeline. A more efficient multi-GPU training & evaluation code is available by contacting zhuohengli@foxmail.com.

## Performance
Static camera reconstruction loss
![](README_md_files/a1458ad0-143b-11ef-9521-4f1cdbadae6e.jpeg?v=1&type=image)
Wrist camera reconstruction loss
![](README_md_files/bfbcc460-143b-11ef-9521-4f1cdbadae6e.jpeg?v=1&type=image)
Arm action loss
![](README_md_files/48c809f0-143b-11ef-9521-4f1cdbadae6e.jpeg?v=1&type=image)
Gripper action loss
![](README_md_files/7fc15f60-143b-11ef-9521-4f1cdbadae6e.jpeg?v=1&type=image)
Average sequence length / success rate (96 evaluations, not accurate enough):
![](README_md_files/ee920e30-143b-11ef-9521-4f1cdbadae6e.jpeg?v=1&type=image)
Learning rate
![](README_md_files/0df1abf0-143c-11ef-9521-4f1cdbadae6e.jpeg?v=1&type=image)

**After the training, I tested the policy with 1000 evaluations. The finetuned checkpoint reaches similar performance (average sequence length 3.199) as the official checkpoint (average sequence length 3.06 in the paper, 3.25 in my evaluation).**

## Notice
- I have not trained it from scratch yet ([according to the authors](https://github.com/bytedance/GR-1/issues/2), the Ego4D pretraining takes 4 days on 32 V100 16GB, and the CALVIN finetuning takes 1 day on 32 V100 16GB). I just finetuned their checkpoint to adapt to my training pipeline. 
- Different from the original paper, I multiply the arm loss with 100 in the total loss, which improves the performance of the policy. I am not sure the actual loss coefficient used in their implementation. See [this issue](https://github.com/bytedance/GR-1/issues/7).
- I use RandomResizedCrop instead of random shifting, but you can try my code [here](https://github.com/bytedance/GR-1/issues/5). RandomResizedCrop seems to have better performance in my experiment.
- I restarted the training around 35k steps so there is fluctuation in the loss and learning rate curves. 

## Acknowledgement
Great thanks to [@bdrhtw](https://github.com/bdrhtw) to make it open-source!

## Contact Me
Email: zhuohengli@foxmail.com

Find Zhuoheng Li in HuggiingFace LeRobot server: [![Discord](https://dcbadge.vercel.app/api/server/C5P34WJ68S?style=flat)](https://discord.gg/s3KuuzsPFb)
(try to merge this repo to LeRobot)

Wechat group for GR1 reproduction: 

![图片](https://github.com/EDiRobotics/GR1-Training/assets/33491471/3c03f32a-d6f3-4990-b2a3-45fe1ab09bb9)

Or feel free to open an issue here.
