# GR1-Training


![](README_md_files/aeb5db90-1cef-11ef-b8f1-596730d257b3.jpeg?v=1&type=image)

Note: Bytedance releases GR-MG, which uses a image diffusion model to generate goal images for GR-1. They also release a pretrained GR-1 trained on Ego4D dataset only. Please see this [link](https://gr-mg.github.io/).

A variant of GR-1: "Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation". It performs good on [CALVIN Manipulation Benchmark](http://calvin.cs.uni-freiburg.de/) without using depth information. The original implementation is [here](https://github.com/bytedance/GR-1) but there is no training script.

**This variant has higher performance than the original implementation on the CALVIN benchmark (current SOTA on ABC->D scenario, I may test ABCD->D later) .** The details, data, multi-GPU training and evaluation code are fully open-source. 

Another nice implementation on CALVIN benchmark is [MDT](https://github.com/intuitive-robots/mdt_policy), which uses a DiT action head + image language alignment loss + masked future image prediction. For differences between MDT and my GR-1 varients, please refer to this [issue](https://github.com/intuitive-robots/mdt_policy/issues/3).

Please remember I build systems for you ヾ(^▽\^*)). Feel free to ask [@StarCycle](https://github.com/StarCycle) if you have any question! 

Also star and cite this repository (and of course the original implementation) if you find it is helpful!

## News
**[2024.7.31]** We also release [mimictest](https://github.com/EDiRobotics/mimictest), which includes RT-1, Diffusion Policy, and Florence policy on the robomimic benchmark. Florence policy is modified from Microsoft's Florence2 VLM, which is trained on 900M images with VQA, detection, segmentation, and OCR tasks. We add a linear action head or a diffusion transformer action head to it. Since Florence policy only contains 0.2/0.7B parameters, it's more light-weight than OpenVLA, RT-2, Roboflamingo, etc. 

**[2024.7.10]** Now it can render the predicted videos. Here are some examples:

| Real | Predicted |
|--|--|
| ![输入图片描述](README_md_files/5c029c40-3e77-11ef-b06d-69973bdd91b6.jpeg?v=1&type=image) | ![输入图片描述](README_md_files/67158a70-3e77-11ef-b06d-69973bdd91b6.jpeg?v=1&type=image) |
| ![输入图片描述](README_md_files/7d6dc170-3e77-11ef-b06d-69973bdd91b6.jpeg?v=1&type=image) | ![](README_md_files/839ab800-3e77-11ef-b06d-69973bdd91b6.jpeg?v=1&type=image) |

To render the videos, please set `without_norm_pixel_loss=true` during training and `record_evaluation_video=true` during inference. Train GR-Chunk with `without_norm_pixel_loss=true` seems to have a lower average length (3.46 on ABC->D, the checkpoint is [here](https://huggingface.co/datasets/StarCycle/calvin_lmdb/blob/main/GR1_chunksize10_withoutpixnorm.pth)).

**[2024.6.17]** Release the initial version of **GR-Diffusion**, which denoises both the predicted images and actions. It can directly load Bytedance's GR-1 checkpoint but its performance is worse than GR-Chunk. 

Please refer to the [grdiffusion](https://github.com/EDiRobotics/GR1-Training/tree/grdiffusion) branch. 

**[2024.5.28]** Release **GR-Chunk** which has higher performance. Specifically, the followings are updated:

 - The actions predicted by GR-Chunk has shape (sequence length, action dim), which improves the average length from 3.25 to 3.556 on CALVIN's ABC->D scenerio (I uploads the log to `evaluation_logs` folder). See the method section. However, you can always load Bytedance's weights and use their settings by modifying `configs.json`.
 - This implementation can be directly used for multi-GPU training and evaluation. I run it on 12*4090 GPUs but it can be easily scaled if you have more computing resources.
 - Unlike the original implementation, GR-Chunk does not have old dependencies like `transformers==4.5.1`. Other dependencies mainly comes from CALVIN so you can discard them if you use other environments.
 - I use the same image shifting approach of the original implementation. The hyper-parameters (except for chunking) are as close as possible.
 -  Add independent evaluation script and modify some APIs.

**[2024.5.16]** Initial data, checkpoint, training & evaluation code released.


## Installation

Setup conda environment and install the CALVIN benchmark. Notice that you do not need this step if you don't want to use CALVIN simulation (so just install my repository).
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
cd ..
```
Install this repository:
```
git clone https://github.com/EDiRobotics/GR1-Training
cd ./GR1-Training
pip install -r requirements.txt
apt-get install -y libegl1-mesa libegl1
apt-get install -y libgl1
apt-get install -y libosmesa6-dev
apt-get install -y patchelf
```

## Prepare Dataset
You do not need to download any datasets if you just want to evaluate the checkpoint. 

If you want to train it, the [original CALVIN dataset](https://github.com/mees/calvin/tree/main/dataset) is too large (~500GB for each task) and contains trajectories data without any language annotation. Here, we only use the trajectories with annotations for training (the same as GR-1 and 3D Diffuser Actor). 

As an example, let's download the CALVIN debug dataset (1.3GB) and transfer it to our LMDB format.
```
wget http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip
unzip calvin_debug_dataset.zip
python calvin2lmdb.py --input_dir ./calvin_debug_dataset --output_dir ./calvin_lmdb
```
You can also download the processed LMDB dataset (ABC->D split) from Huggingface. The LMDB dataset only takes ~23GB, while the original ABC->D split takes 517GB. In this example, I use the tool of [HF-Mirror](https://hf-mirror.com/). You can set the environment variable `export HF_ENDPOINT=https://hf-mirror.com` to avoid the connection problem in some regions.
```
rm -rf calvin_lmdb
apt install git-lfs aria2
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
./hfd.sh StarCycle/calvin_lmdb --dataset --tool aria2c -x 4
```
## Config HuggingFace Accelerate & Setup CALVIN Simulation

To config accelerate, run this command and follow its guidance. I use `bf16` in training.
```
accelerate config
```

To setup CALVIN, use
```
export CALVIN_ROOT=<path to calvin folder>
```

## Prepare Weights
You need to download the weights of the ViT visual encoder. The weights of ViT is [here](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth). The model will also automatically download weights of CLIP when loading it.

For weights of policy, you can use the [Bytedance's weights for CALVIN's ABC->D](https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/gr1_code_release/snapshot_ABC.pt) or my weights. My weights (chunk size==1 or 10) are within the HuggingFace dataset mentioned above.

If you choose to use Bytedance's weights, please set the followings in `configs.json`. My implementation is always compatible with Bytedance's weights no matter what chunk size you choose.
```
"bytedance_ckpt_path": "<path to the weight>",
"load_bytedance_ckpt": true,
"mae_ckpt": "<path to ViT weight>",
```

If you choose to use my weights, please set
```
"load_bytedance_ckpt": false,
"load_epoch": <the epoch of checkpoint you want>,
"save_path": "<the folder you save the checkpoints & log>",
"mae_ckpt": "<path to ViT weight>",
```
It loads `GR1_<the epoch you select>.pth` from `./Save/`.  Notice that:
- To use my weights, copy it from the downloaded lmdb dataset and rename it as `GR1_<load_epoch>.pth`. 
- My weights with different `chunk_size` are incompatible, but you can slightly modify `state_dict['action_chunk_queries.weight']` to solve it.
- `GR1_0_chunksize1.pth` (197MB) is trained with transformers 4.11, which will save `attn.bias` and `attn.mask_bias`. By contrast, `GR1_0_chunksize10.pth` (186MB) is trained with newest transformers, which will not save such parameters. 

## Evaluation

Remember to set these in `configs.json` (you may not use my hyper-parameters):
```
"record_evaluation_video": "<true or false>",
"bytedance_ckpt_path": "<path to the Bytedance's weight>",
"load_bytedance_ckpt": <true or false>,
"load_epoch": <the epoch of checkpoint you want>,
"num_sequences": <how many episodes you want to simulate, I use 1000>,
"ep_len": <maximum step number of a task in an episode>,
"chunk_size": <action chunk size of the network>,
"test_chunk_size": <the action chunk size you actually execute>,
```

Then simply run
```
accelerate launch evaluate_calvin.py
```
It will run N simulations in parallel on N GPUs (depending on what you set in `accelerate config`). If you choose to record the evaluation video, the videos will be saved in `eval_<GPU id>` folder under the `save_path` you specify.

**If you have EGL related errors**: this error can usually be fixed by install some packages depending on your system. I recommend to use `conda install -c conda-forge <package>` instead of `apt install` since it usually fixes my error. You can contact me if it's still not solved

## Training

After setting `configs.json`, you can simply launch training with 
```
accelerate launch Main.py
```
If `"evaluate_during_training": true` in `configs.json`, then it will evaluate the policy after every epoch of training.

## GR-Chunk Method

Let's assume `sequence_len==4`, `chunk_size==3`, `test_chunk_size==2` and the currect timestep is 4, the input and output of the original GR1 is simply (s1, s2, s3, s4) and (a1, a2, a3, a4), respectively. In the environment, we take the predicted action a4.

By contrast, the output in GR-Chunk is ((a1, a2, a3), (a2, a3, a4), (a3, a4, a5), (a4, a5, a6)). In the environment, we take the predicted action a4 and a5 in consecutive timesteps, and then run the policy to predict future actions again. 

Similar approach is taken in  Meta's recent [4-token prediction LLM](https://arxiv.org/pdf/2404.19737), [ACT/Aloha](https://github.com/tonyzhaozh/act) of @[tonyzhaozh](https://github.com/tonyzhaozh/act/commits?author=tonyzhaozh), [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) of @[cheng-chi](https://github.com/real-stanford/diffusion_policy/commits?author=cheng-chi), and [Octo](https://github.com/octo-models/octo).

In my experiment, temporal ensembling of ACT does not improve the success rate.  Octo has similar conclusion. As reported by @[tonyzhaozh](https://github.com/tonyzhaozh/act/commits?author=tonyzhaozh), temporal ensembling seems to work better with small data.

The best `test_chunk_size` seems to be 1 when `chunk_size==10`, see the following ablation:

| Configuration | Avg. Len of ABC->D |
|--|--|
| chunk_size=1, test_chunk_size=1 | 3.257 |
| chunk_size=10, test_chunk_size=1 | 3.556 |
| chunk_size=10, test_chunk_size=2 | 2.145 |

## Detailed Analysis of the Original Network and API
Please refer to this [issue](https://github.com/bytedance/GR-1/issues/4).

## Training Curves

I first finetune the policy with `chunk_size==1` and then keep finetuning it with `chunk_size==10`. Each phase takes 20 epochs. I guess you can directly train it with `chunk_size==10`.

I do not evaluate the policy during training because the server I use now does not have Nvidia EGL support.

First finetune it with `chunk_size==1` (8*4090 GPU, batch size per GPU 22, gradient accumulation step 3):

Static camera reconstruction loss
![](README_md_files/5fcbb420-1cf7-11ef-b8f1-596730d257b3.jpeg?v=1&type=image)

Wrist camera reconstruction loss
![](README_md_files/4ffc0130-1cf7-11ef-b8f1-596730d257b3.jpeg?v=1&type=image)

Arm action loss
![](README_md_files/0b7854f0-1cf7-11ef-b8f1-596730d257b3.jpeg?v=1&type=image)

Gripper action loss
![](README_md_files/26b6d1b0-1cf7-11ef-b8f1-596730d257b3.jpeg?v=1&type=image)

Learning rate
![](README_md_files/3f840050-1cf7-11ef-b8f1-596730d257b3.jpeg?v=1&type=image)

First finetune it with `chunk_size==10` (8*4090 GPU, batch size per GPU 19, gradient accumulation step 3):

Static camera reconstruction loss
![](README_md_files/b7bc1030-1cf7-11ef-b8f1-596730d257b3.jpeg?v=1&type=image)

Wrist camera reconstruction loss
![](README_md_files/caabc7d0-1cf7-11ef-b8f1-596730d257b3.jpeg?v=1&type=image)

Arm action loss
![](README_md_files/e53350a0-1cf7-11ef-b8f1-596730d257b3.jpeg?v=1&type=image)

Gripper action loss
![](README_md_files/f9942470-1cf7-11ef-b8f1-596730d257b3.jpeg?v=1&type=image)

Learning rate
![](README_md_files/0de42f60-1cf8-11ef-b8f1-596730d257b3.jpeg?v=1&type=image)

## Notice
- I have not trained it from scratch yet ([according to the authors](https://github.com/bytedance/GR-1/issues/2), the Ego4D pretraining takes 4 days on 32 V100 16GB, and the CALVIN finetuning takes 1 day on 32 V100 16GB). I just finetuned their checkpoint to adapt to my training pipeline. 
- When I measure the success rate of the Bytedance's weight, I get an average length of 3.25 instead of 3.06 in their original paper. Perhaps they did not fix this [issue](https://github.com/mees/calvin/issues/32#issuecomment-1363352121) in CALVIN benchmark, while 3D Diffuser Actor fixed it

## Acknowledgement
Great thanks to [@bdrhtw](https://github.com/bdrhtw) to make it open-source! 

## Feel Free to Contact Me!


Email: zhuohengli@foxmail.com

Find Zhuoheng Li in HuggiingFace LeRobot server: [![Discord](https://dcbadge.vercel.app/api/server/C5P34WJ68S?style=flat)](https://discord.gg/s3KuuzsPFb)
(try to merge this repo to LeRobot)

Wechat group: 

![图片](https://github.com/EDiRobotics/GR1-Training/assets/33491471/1250bcc6-52a8-4da0-89d2-c8ed55bb4613)

Or feel free to open an issue here.
