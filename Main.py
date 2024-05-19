import os
import math
import json
from time import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import clip
from LMDBDataset_jpeg import LMDBDataset as LMDBdst_jpeg
from LMDBDataset_jpeg import DataPrefetcher as DataPrefetcher_jpeg
from PreProcess import PreProcess
import models.vision_transformer as vits
from models.gr1 import GR1 
from evaluate_calvin import make_env, evaluate_policy 
from evaluation.calvin_evaluation import GR1CalvinEvaluation 

def masked_loss(pred, target, mask, skip_frame=0, loss_func=F.mse_loss):
    if skip_frame == 0:
        new_pred = pred
    else:
        new_pred = pred[:, :-skip_frame]
    new_target = target[:, skip_frame:]
    new_mask = mask[:, skip_frame:]
    data_shape, mask_shape = new_pred.shape, new_mask.shape
    loss = loss_func(new_pred, new_target, reduction='none')
    for _ in range(len(data_shape) - len(mask_shape)):
        new_mask = new_mask.unsqueeze(-1)
    loss = (loss*new_mask).sum() / new_mask.sum() / math.prod(data_shape[len(mask_shape):])
    return loss

def train(train_prefetcher, test_prefetcher, preprocessor, model, env, eva, eval_dir, optimizer, scheduler, scaler, device, cfg, step, writer):
    train_dataset_len = len(train_prefetcher.loader.dataset)
    test_dataset_len = len(test_prefetcher.loader.dataset)
    eval_steps = train_dataset_len // test_dataset_len
    avg_reward = 0.0
    for epoch in range(cfg['num_epochs']):
        if epoch != 0 and epoch % cfg['save_epochs'] == 0:
            model.eval()
            avg_reward = torch.tensor(evaluate_policy(
                eva, 
                env,
                cfg['save_path']+'success_rate.txt', 
                cfg['save_path']+'result.txt', 
                cfg['eval_num_sequences'], 
                cfg['ep_len'], 
                eval_dir,
                debug=False
            )).float().mean().to(device)
            modules_to_exclude = ['model_mae', 'model_clip']
            state_dict = {k: v for k, v in model.state_dict().items() if not any(module_name in k for module_name in modules_to_exclude)}
            torch.save({'state_dict': state_dict}, cfg['save_path']+'GR1_{}.pth'.format(epoch+cfg['load_epoch']))

        log_loss = {
            'rgb_static': 0,
            'rgb_gripper': 0,
            'action_arm': 0,
            'action_gripper': 0,
        }
        eval_log_loss = {
            'rgb_static': 0,
            'rgb_gripper': 0,
            'action_arm': 0,
            'action_gripper': 0,
        }
        for key in log_loss:
            log_loss[key] = torch.tensor(0).float().to(device)
        for key in eval_log_loss:
            eval_log_loss[key] = torch.tensor(0).float().to(device)
        cum_load_time = 0 
        clock = time()
        batch_idx = 0
        batch, load_time = train_prefetcher.next()
        while batch is not None:
            model.train()
            with autocast(dtype=torch.bfloat16):
                rgb_static, rgb_gripper = preprocessor.rgb_process(batch['rgb_static'], batch['rgb_gripper'], train=True)
                pred = model(
                    rgb=rgb_static,
                    hand_rgb=rgb_gripper,
                    state={'arm': batch['arm_state'], 'gripper': batch['gripper_state']},
                    language=batch['inst_token'],
                    attention_mask=batch['mask'],
                )
                loss = {}
                loss['rgb_static'] = masked_loss(pred['obs_preds'], pred['obs_targets'], batch['mask'], cfg['skip_frame'], F.mse_loss)
                loss['rgb_gripper'] = masked_loss(pred['obs_hand_preds'], pred['obs_hand_targets'], batch['mask'], cfg['skip_frame'], F.mse_loss)
                loss['action_arm'] = masked_loss(pred[ 'arm_action_preds'], batch['actions'][:, :, :6], batch['mask'], 0, F.smooth_l1_loss)
                loss['action_gripper'] = masked_loss(pred['gripper_action_preds'], batch['actions'][:, :, -1:], batch['mask'], 0, F.binary_cross_entropy_with_logits)
                total_loss = loss['rgb_static'] + loss['rgb_gripper'] +100*loss['action_arm'] + loss['action_gripper'] # TODO
            scaler.scale(total_loss / cfg['gradient_accumulation_steps']).backward()
            if (batch_idx + 1) % cfg['gradient_accumulation_steps'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            for key in log_loss:
                log_loss[key] += loss[key].detach() / cfg['print_steps']
            cum_load_time += load_time / cfg['print_steps']

            if batch_idx % eval_steps == 0:
                with torch.no_grad():
                    model.eval()
                    batch, _ = test_prefetcher.next_without_none()
                    rgb_static, rgb_gripper = preprocessor.rgb_process(batch['rgb_static'], batch['rgb_gripper'], train=False)
                    pred = model(
                        rgb=rgb_static,
                        hand_rgb=rgb_gripper,
                        state={'arm': batch['arm_state'], 'gripper': batch['gripper_state']},
                        language=batch['inst_token'],
                        attention_mask=batch['mask'],
                    )
                    loss = {}
                    loss['rgb_static'] = masked_loss(pred['obs_preds'], pred['obs_targets'], batch['mask'], cfg['skip_frame'], F.mse_loss)
                    loss['rgb_gripper'] = masked_loss(pred['obs_hand_preds'], pred['obs_hand_targets'], batch['mask'], cfg['skip_frame'], F.mse_loss)
                    loss['action_arm'] = masked_loss(pred[ 'arm_action_preds'], batch['actions'][:, :, :6], batch['mask'], 0, F.smooth_l1_loss)
                    loss['action_gripper'] = masked_loss(pred['gripper_action_preds'], batch['actions'][:, :, -1:], batch['mask'], 0, F.binary_cross_entropy_with_logits)
                    for key in eval_log_loss:
                        eval_log_loss[key] += loss[key].detach() / cfg['print_steps'] * eval_steps

            if batch_idx % cfg['print_steps'] == 0 and batch_idx != 0:
                load_pecnt = torch.tensor(cum_load_time / (time()-clock)).to(device)
                fps = (cfg['bs_per_gpu']*cfg['print_steps']*cfg['seq_len']) / (time()-clock)

                text = 'Train Epoch: {} [{}/{} ({:.0f}%)] Reward: {:.5f} FPS:{:.5f} Load Pertentage:{:.5f} LR:{}'.format(
                    epoch, 
                    batch_idx * cfg['bs_per_gpu'], 
                    train_dataset_len, 
                    100. * batch_idx * cfg['bs_per_gpu'] / train_dataset_len, 
                    avg_reward,
                    fps,
                    load_pecnt,
                    scheduler.get_last_lr()[0],
                )
                for key in log_loss:
                    text = text + ' {}_loss: {:.5f}'.format(key, log_loss[key])
                for key in eval_log_loss:
                    text = text + ' eval_{}_loss: {:.5f}'.format(key, eval_log_loss[key])
                print(text)
                for key in log_loss:
                    writer.add_scalar(key+'_loss', log_loss[key], step)
                for key in eval_log_loss:
                    writer.add_scalar('eval_'+key+'_loss', eval_log_loss[key], step)
                writer.add_scalar("reward", avg_reward, step)
                writer.add_scalar("learning rate", scheduler.get_last_lr()[0], step)
                writer.add_scalar("FPS", fps, step)
                writer.add_scalar("loading time in total time", load_pecnt, step)
                with open(cfg['save_path']+'step.json', 'w') as json_file:
                    json.dump(step, json_file)

                for key in log_loss:
                    log_loss[key] = torch.tensor(0).float().to(device)
                for key in eval_log_loss:
                    eval_log_loss[key] = torch.tensor(0).float().to(device)
                cum_load_time = 0
                clock = time()
                scheduler.step()

            batch_idx += 1
            step += 1
            batch, load_time = train_prefetcher.next()

if __name__ == '__main__':
    # Preparation
    cfg = json.load(open('configs.json'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocessor = PreProcess(
        cfg['rgb_shape'],
        cfg['rgb_mean'],
        cfg['rgb_std'],
        cfg['crop_area_scale'],
        cfg['crop_aspect_ratio'],
        device,
    )
    train_dataset = LMDBdst_jpeg(
        cfg['LMDB_path'], 
        cfg['seq_len'], 
        cfg['action_mode'],
        cfg['act_dim'],
        start_ratio = 0,
        end_ratio = 0.9, 
    )
    test_dataset = LMDBdst_jpeg(
        cfg['LMDB_path'], 
        cfg['seq_len'], 
        cfg['action_mode'],
        cfg['act_dim'],
        start_ratio = 0.9,
        end_ratio = 1, 
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['bs_per_gpu'], # to be flattened in prefetcher  
        num_workers=cfg['workers_per_gpu'],
        pin_memory=True, # Accelerate data reading
        shuffle=True,
        prefetch_factor=cfg['prefetch_factor'],
        persistent_workers=True,
    ) 
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg['bs_per_gpu'], # to be flattened in prefetcher  
        num_workers=cfg['workers_per_gpu'],
        pin_memory=True, # Accelerate data reading
        shuffle=True,
        prefetch_factor=cfg['prefetch_factor'],
        persistent_workers=True,
    ) 
    model_clip, _ = clip.load(cfg['clip_backbone'], device=device) 
    model_mae = vits.__dict__['vit_base'](patch_size=16, num_classes=0).to(device)
    checkpoint = torch.load(cfg['mae_ckpt'])
    model_mae.load_state_dict(checkpoint['model'], strict=False)
    model = GR1(
        model_clip,
        model_mae,
        state_dim=cfg['state_dim'],
        act_dim=cfg['act_dim'],
        hidden_size=cfg['embed_dim'],
        sequence_length=cfg['seq_len'],
        training_target=['act_pred', 'fwd_pred', 'fwd_pred_hand'],
        img_feat_dim=cfg['img_feat_dim'],
        patch_feat_dim=cfg['patch_feat_dim'],
        lang_feat_dim=cfg['lang_feat_dim'],
        resampler_params={
            'depth': cfg['resampler_depth'],
            'dim_head': cfg['resampler_dim_head'],
            'heads': cfg['resampler_heads'],
            'num_latents': cfg['resampler_num_latents'],
            'num_media_embeds': cfg['resampler_num_media_embeds'],
        },
        without_norm_pixel_loss=False,
        use_hand_rgb=True,
        n_layer=cfg['n_layer'],
        n_head=cfg['n_head'],
        n_inner=4*cfg['embed_dim'],
        activation_function=cfg['activation_function'],
        n_positions=cfg['n_positions'],
        resid_pdrop=cfg['dropout'],
        attn_pdrop=cfg['dropout'],
    ).to(device)  # for fused optimizer
    if os.path.isfile(cfg['save_path']+'GR1_{}.pth'.format(cfg['load_epoch'])):
        model.load_state_dict(torch.load(cfg['save_path']+'GR1_{}.pth'.format(cfg['load_epoch']))['state_dict'], strict=False)
        print('load model')
    if os.path.isfile(cfg['save_path']+'step.json'):
        with open(cfg['save_path']+'step.json', 'r') as json_file:
            step = json.load(open(cfg['save_path']+'step.json'))
    else:
        step = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr_max'], weight_decay=cfg['weight_decay'], fused=True)
    total_prints_per_epoch = len(train_dataset) // (cfg['print_steps'] * cfg['bs_per_gpu'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=cfg['num_warmup_epochs']*total_prints_per_epoch,
        num_training_steps=cfg['num_epochs']*total_prints_per_epoch,
    )
    scaler = GradScaler()
    train_prefetcher = DataPrefetcher_jpeg(train_loader, device)
    test_prefetcher = DataPrefetcher_jpeg(test_loader, device)
    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper'], 
        'depth_obs': [], 
        'state_obs': ['robot_obs'], 
        'actions': ['rel_actions'], 
        'language': ['language']}
    env = make_env('./fake_dataset', observation_space, device)
    eval_dir = cfg['save_path']+f'eval{torch.cuda.current_device()}/'
    os.makedirs(eval_dir, exist_ok=True)
    eva = GR1CalvinEvaluation(model, cfg, device)
    writer = SummaryWriter(cfg['save_path'] + 'logs')

    # Train
    train(train_prefetcher, test_prefetcher, preprocessor, model, env, eva, eval_dir, optimizer, scheduler, scaler, device, cfg, step, writer)
