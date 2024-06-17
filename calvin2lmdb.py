import os
import io
import argparse
import lmdb
from pickle import dumps, loads
import numpy as np
import torch
from torchvision.transforms.functional import resize
from torchvision.io import encode_jpeg
import clip
from einops import rearrange, repeat

def save_to_lmdb(output_dir, input_dir):
    env = lmdb.open(output_dir, map_size=int(3e12), readonly=False, lock=False) # maximum size of memory map is 3TB
    annotations = np.load(os.path.join(input_dir, 'lang_annotations/auto_lang_ann.npy'), allow_pickle=True).tolist()['language']['ann']
    start_end_ids = np.load(os.path.join(input_dir, 'lang_annotations/auto_lang_ann.npy'), allow_pickle=True).tolist()['info']['indx']
    with env.begin(write=True) as txn:
        if txn.get('cur_step'.encode()) is not None:
            cur_step = loads(txn.get('cur_step'.encode())) + 1
            cur_episode = loads(txn.get(f'cur_episode_{cur_step - 1}'.encode())) + 1
        else:
            cur_step = 0
            cur_episode = 0

        for index, (start, end) in enumerate(start_end_ids):
            print(f'{index/len(start_end_ids)}')
            inst = annotations[index]
            txn.put(f'inst_{cur_episode}'.encode(), dumps(inst))
            with torch.no_grad():
                inst_token = clip.tokenize(inst)
                inst_emb = model_clip.encode_text(inst_token.cuda()).cpu()
            txn.put(f'inst_token_{cur_episode}'.encode(), dumps(inst_token[0]))
            txn.put(f'inst_emb_{cur_episode}'.encode(), dumps(inst_emb[0]))
            for i in range(start, end+1):
                frame = np.load(os.path.join(input_dir, f'episode_{i:07}.npz'))
                txn.put('cur_step'.encode(), dumps(cur_step))
                txn.put(f'cur_episode_{cur_step}'.encode(), dumps(cur_episode))
                txn.put(f'done_{cur_step}'.encode(), dumps(False))
                rgb_static = torch.from_numpy(rearrange(frame['rgb_static'], 'h w c -> c h w'))
                txn.put(f'rgb_static_{cur_step}'.encode(), dumps(encode_jpeg(rgb_static)))
                rgb_gripper = torch.from_numpy(rearrange(frame['rgb_gripper'], 'h w c -> c h w'))
                txn.put(f'rgb_gripper_{cur_step}'.encode(), dumps(encode_jpeg(rgb_gripper)))
                txn.put(f'abs_action_{cur_step}'.encode(), dumps(torch.from_numpy(frame['actions'])))
                txn.put(f'rel_action_{cur_step}'.encode(), dumps(torch.from_numpy(frame['rel_actions'])))
                txn.put(f'scene_obs_{cur_step}'.encode(), dumps(torch.from_numpy(frame['scene_obs'])))
                txn.put(f'robot_obs_{cur_step}'.encode(), dumps(torch.from_numpy(frame['robot_obs'])))
                cur_step += 1
            txn.put(f'done_{cur_step-1}'.encode(), dumps(True))
            cur_episode += 1
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transfer CALVIN dataset to lmdb format.")
    parser.add_argument("--input_dir", default='./ABC_D', type=str, help="Original dataset directory.")
    parser.add_argument("--output_dir", default='./calvin_lmdb', type=str, help="Original dataset directory.")
    args = parser.parse_args()
    model_clip, _ = clip.load('ViT-B/32', device='cuda:0')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_to_lmdb(args.output_dir, os.path.join(args.input_dir, 'training'))
    save_to_lmdb(args.output_dir, os.path.join(args.input_dir, 'validation'))