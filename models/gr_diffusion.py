# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GR-1 model."""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import GPT2Model
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

from flamingo_pytorch import PerceiverResampler
from models.vision_transformer import Block
from models.transformer_utils import get_2d_sincos_pos_embed

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

class DiffusionPolicy():
    def __init__(self,
            model_clip,
            model_mae,
            state_dim,
            act_dim,
            hidden_size,
            sequence_length,
            chunk_size,
            training_target,
            img_feat_dim,
            patch_feat_dim,
            lang_feat_dim,
            resampler_params,
            n_layer,
            n_head,
            n_inner,
            activation_function,
            n_positions,
            resid_pdrop,
            attn_pdrop,
            without_norm_pixel_loss,
            use_hand_rgb,
            num_train_steps,
            num_infer_steps,
            device
        ):
        
        self.net = GR_Diffusion(
            model_clip,
            model_mae,
            state_dim,
            act_dim,
            hidden_size,
            sequence_length,
            chunk_size,
            training_target,
            img_feat_dim,
            patch_feat_dim,
            lang_feat_dim,
            resampler_params,
            n_layer,
            n_head,
            n_inner,
            activation_function,
            n_positions,
            resid_pdrop,
            attn_pdrop,
            without_norm_pixel_loss,
            use_hand_rgb,
        ).to(device)
        self.ema = EMAModel(
            parameters=self.net.parameters(),
            power=0.75)
        self.ema_net = copy.deepcopy(self.net) 

        self.chunk_size = chunk_size
        self.act_dim = act_dim
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=False, # TODO
            prediction_type='sample',
            set_alpha_to_one=True,
            steps_offset=0,
        )
        self.noise_scheduler.set_timesteps(num_infer_steps)
        self.device = device

    def compute_loss(self, state, language, rgb, hand_rgb, obs_mask, skip_rgb_frame, actions, action_mask, action_loss_ratio):
        # sample a diffusion iteration for each data point
        B = rgb.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()

        encoded_feat = self.net(
            mode = 'encode',
            rgb = rgb, 
            hand_rgb = hand_rgb, 
            state = state, 
            language = language,
        )

        # add noise to the clean action/obs according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        action_noise = torch.randn(actions.shape, device=self.device)
        obs_noise = torch.randn(encoded_feat['obs_targets'].shape, device=self.device)
        hand_obs_noise = torch.randn(encoded_feat['obs_hand_targets'].shape, device=self.device)
        noisy_actions = self.noise_scheduler.add_noise(actions, action_noise, timesteps)
        noisy_obs = self.noise_scheduler.add_noise(encoded_feat['obs_targets'], obs_noise, timesteps)
        noisy_hand_obs = self.noise_scheduler.add_noise(encoded_feat['obs_hand_targets'], hand_obs_noise, timesteps)
        noisy_actions = self.noise_scheduler.scale_model_input(noisy_actions, timesteps)
        noisy_obs = self.noise_scheduler.scale_model_input(noisy_obs, timesteps)
        noisy_hand_obs = self.noise_scheduler.scale_model_input(noisy_hand_obs, timesteps)

        pred = self.net(
            mode = 'denoise',
            stacked_inputs = encoded_feat['stacked_inputs'],
            noisy_obs = noisy_obs,
            noisy_hand_obs = noisy_hand_obs,
            noisy_actions = noisy_actions,
            diffusion_step = timesteps,
            attention_mask = obs_mask,
        )

        loss = {}
        loss['rgb_static'] = masked_loss(pred['obs_preds'], encoded_feat['obs_targets'], obs_mask, skip_rgb_frame, F.mse_loss)
        loss['rgb_gripper'] = masked_loss(pred['obs_hand_preds'], encoded_feat['obs_hand_targets'], obs_mask, skip_rgb_frame, F.mse_loss)
        pred_actions = torch.cat((pred['arm_action_preds'], F.sigmoid(pred['gripper_action_preds'])), dim=-1)
        loss['action'] = masked_loss(pred_actions, actions, action_mask, 0, F.smooth_l1_loss)
        loss['total_loss'] = loss['rgb_static'] + loss['rgb_gripper'] + action_loss_ratio*loss['action']
        return loss 

    def infer(self, rgb, hand_rgb, state, language, attention_mask):
        B, T, _, _, _ = rgb.shape
        encoded_feat = self.ema_net(
            mode = 'encode',
            rgb = rgb, 
            hand_rgb = hand_rgb, 
            state = state, 
            language = language,
        )
        noisy_obs = torch.randn(encoded_feat['obs_targets'].shape, device=self.device)
        noisy_hand_obs = torch.randn(encoded_feat['obs_hand_targets'].shape, device=self.device)
        noisy_actions = torch.randn((B, T, self.chunk_size, self.act_dim), device=self.device)
        noisy_actions = self.noise_scheduler.scale_model_input(noisy_actions, timesteps)
        noisy_obs = self.noise_scheduler.scale_model_input(noisy_obs, timesteps)
        noisy_hand_obs = self.noise_scheduler.scale_model_input(noisy_hand_obs, timesteps)
        for k in self.noise_scheduler.timesteps:
            timesteps = torch.ones((B,), dtype=torch.long, device=self.device) * k
            pred = self.ema_net(
                mode = 'denoise',
                stacked_inputs = encoded_feat['stacked_inputs'],
                noisy_obs = noisy_obs,
                noisy_hand_obs = noisy_hand_obs,
                noisy_actions = noisy_actions,
                diffusion_step = timesteps,
                attention_mask = attention_mask,
            )
            noisy_obs = self.noise_scheduler.step(
                model_output=pred['obs_preds'],
                timestep=k,
                sample=noisy_obs,
            ).prev_sample
            noisy_hand_obs = self.noise_scheduler.step(
                model_output=pred['obs_hand_preds'],
                timestep=k,
                sample=noisy_hand_obs,
            ).prev_sample
            pred_actions = torch.cat((pred['arm_action_preds'], pred['gripper_action_preds']), dim=-1)
            noisy_actions = self.noise_scheduler.step(
                model_output=pred_actions,
                timestep=k,
                sample=noisy_actions,
            ).prev_sample
        return noisy_actions
                        
    def update_ema(self):
        self.ema.step(self.net.parameters())

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class GR_Diffusion(nn.Module):
    def __init__(
            self,
            model_clip,
            model_mae,
            state_dim,
            act_dim,
            hidden_size,
            sequence_length,
            chunk_size,
            training_target,
            img_feat_dim,
            patch_feat_dim,
            lang_feat_dim,
            resampler_params,
            n_layer,
            n_head,
            n_inner,
            activation_function,
            n_positions,
            resid_pdrop,
            attn_pdrop,
            without_norm_pixel_loss=False,
            use_hand_rgb=True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size

        # GPT
        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            n_positions=n_positions,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
        )
        self.transformer = GPT2Model(config)

        # Perciever resampler
        self.n_patch_latents = resampler_params['num_latents']
        self.perceiver_resampler = PerceiverResampler(
            dim=patch_feat_dim,
            depth=resampler_params['depth'],
            dim_head=resampler_params['dim_head'],
            heads=resampler_params['heads'],
            num_latents=self.n_patch_latents,
            num_media_embeds=resampler_params['num_media_embeds'])        

        # CLIP for language encoding
        self.model_clip = model_clip
        for _, param in self.model_clip.named_parameters():
            param.requires_grad = False

        # MAE for image encoding
        self.model_mae = model_mae
        for _, param in self.model_mae.named_parameters():
            param.requires_grad = False
        
        self.n_patches = 49
        self.patch_size = 16
        self.image_size = 224
        self.img_feat_dim = img_feat_dim
        self.lang_feat_dim = lang_feat_dim
        self.patch_feat_dim = patch_feat_dim
        self.use_hand_rgb = use_hand_rgb

        self.act_pred = False
        self.fwd_pred = False
        self.fwd_pred_hand = False
        if 'act_pred' in training_target:
            self.act_pred = True
        if 'fwd_pred' in training_target:
            self.fwd_pred = True
        if 'fwd_pred_hand' in training_target:
            self.fwd_pred_hand = True
        
        self.without_norm_pixel_loss = without_norm_pixel_loss

        # Embedding functions for states
        self.embed_arm_state = torch.nn.Linear(self.state_dim - 1, hidden_size)
        self.embed_gripper_state = torch.nn.Linear(2, hidden_size) # one-hot gripper state
        self.embed_state = torch.nn.Linear(2*hidden_size, hidden_size)

        # Relative timestep embedding
        self.embed_timestep = nn.Embedding(self.sequence_length, hidden_size)

        # Embedding function for languages
        self.embed_lang = torch.nn.Linear(self.lang_feat_dim, hidden_size)

        # Embedding functions for images
        self.embed_hand_img = torch.nn.Linear(self.img_feat_dim, hidden_size)
        self.embed_img = torch.nn.Linear(self.img_feat_dim, hidden_size)
        self.embed_hand_patch = torch.nn.Linear(self.patch_feat_dim, hidden_size) 
        self.embed_patch = torch.nn.Linear(self.patch_feat_dim, hidden_size)

        # Layer norm
        self.embed_ln = nn.LayerNorm(hidden_size)

        # Action query token
        self.action_queries = nn.Embedding(chunk_size, hidden_size) # arm + gripper, weight from bytedance

        # Noisy action encoder
        self.embed_noisy_actions = nn.Linear(self.act_dim, hidden_size)
        self.embed_noisy_actions.weight.data.fill_(0) # finetune it from zero weight

        # Observation query token
        self.obs_queries = nn.Embedding(self.n_patch_latents + 1, self.hidden_size)
        self.obs_hand_queries = nn.Embedding(self.n_patch_latents + 1, self.hidden_size)

        # Noisy observation encoder
        self.embed_noisy_obs = nn.Linear(3*self.patch_size**2, hidden_size)
        self.embed_noisy_obs.weight.data.fill_(0) # finetune it from zero weight

        # Diffusion step encoder
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.Mish(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )

        # Action prediction
        self.pred_act_mlps = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size//2),
            nn.Linear(hidden_size//2, hidden_size//2)])
        self.pred_arm_act = nn.Linear(hidden_size//2, self.act_dim-1) # arm action
        self.pred_gripper_act = nn.Linear(hidden_size//2, 1) # gripper action (binary)
        
        # Forward prediction
        self.decoder_embed = nn.Linear(hidden_size, hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_size))
        decoder_depth = 2
        self.decoder_blocks = nn.ModuleList([
            Block(hidden_size, 16, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(hidden_size)
        self.decoder_pred = nn.Linear(hidden_size, self.patch_size**2 * 3, bias=True) # decoder to patch
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, (self.image_size//self.patch_size)**2,
            hidden_size), requires_grad=False)  # (1, n_patch, h)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (self.image_size//self.patch_size))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

    def forward(self, 
                mode,
                rgb = None, 
                hand_rgb = None, 
                state = None, 
                language = None, 
                stacked_inputs = None,
                noisy_obs = None,
                noisy_hand_obs = None,
                noisy_actions = None,
                diffusion_step = None,
                attention_mask = None,
    ):
        '''
        When mode == 'encode':
            
            Input:
                rgb, hand_rgb, state, language
            Output:
                stacked_inputs, obs_targets, obs_hand_targets

        When mode == 'denoise':

            Input:
                stacked_inputs, noisy_obs, noisy_hand_obs, noisy_actions, diffusion_step, attention_mask
            Output:
                obs_preds, obs_hand_preds, arm_action_preds, gripper_action_preds,
                
        '''

        if mode == 'encode': # Encode inputs

            obs_targets = None
            obs_hand_targets = None
            batch_size, sequence_length, c, h, w = rgb.shape
            
            # Embed state
            arm_state = state['arm']
            gripper_state = state['gripper']
            arm_state_embeddings = self.embed_arm_state(arm_state.view(batch_size, sequence_length, self.state_dim-1))
            gripper_state_embeddings = self.embed_gripper_state(gripper_state)
            state_embeddings = torch.cat((arm_state_embeddings, gripper_state_embeddings), dim=2)
            state_embeddings = self.embed_state(state_embeddings)  # (b, t, h)

            # Embed language
            lang_embeddings = self.model_clip.encode_text(language)
            lang_embeddings = lang_embeddings / (lang_embeddings.norm(dim=1, keepdim=True) + 1e-6) # normalization 
            lang_embeddings = self.embed_lang(lang_embeddings.float())  # (b, h)
        
            # Get obs and patch feature from MAE
            obs_embeddings, patch_embeddings = self.model_mae(
                rgb.view(batch_size*sequence_length, c, h, w))  # (b * t, img_feat_dim), (b * t, n_patches, patch_feat_dim)
            obs_embeddings = obs_embeddings.view(batch_size, sequence_length, -1)  # (b, t, img_feat_dim)
            if self.use_hand_rgb:
                hand_obs_embeddings, hand_patch_embeddings = self.model_mae(
                    hand_rgb.view(batch_size*sequence_length, c, h, w))
                hand_obs_embeddings = hand_obs_embeddings.view(batch_size, sequence_length, -1)  # (b, t, img_feat_dim)
            if self.fwd_pred:
                p = self.patch_size
                h_p = h // p
                w_p = w // p
                rgb = rgb.reshape(shape=(batch_size, sequence_length, 3, h_p, p, w_p, p)) 
                obs_targets = rgb.permute(0, 1, 3, 5, 4, 6, 2)
                obs_targets = obs_targets.reshape(shape=(batch_size, sequence_length, h_p * w_p, (p**2) * 3))  # (b, t, n_patches, p*p*3)
                if not self.without_norm_pixel_loss:
                    # norm the target 
                    obs_targets = (obs_targets - obs_targets.mean(dim=-1, keepdim=True)
                        ) / (obs_targets.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)
                if self.fwd_pred_hand:
                    hand_rgb = hand_rgb.reshape(shape=(batch_size, sequence_length, 3, h_p, p, w_p, p))
                    obs_hand_targets = hand_rgb.permute(0, 1, 3, 5, 4, 6, 2)
                    obs_hand_targets = obs_hand_targets.reshape(shape=(batch_size, sequence_length, h_p * w_p, (p**2)*3))  # (b, t, n_patches, p*p*3)
                    if not self.without_norm_pixel_loss:
                        # norm the target 
                        obs_hand_targets = (obs_hand_targets - obs_hand_targets.mean(dim=-1, keepdim=True)
                            ) / (obs_hand_targets.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)            

            # Use resampler to process patch embeddings
            patch_embeddings = patch_embeddings.unsqueeze(1)  # (b * t, 1, n_patches, patch_feat_dim)
            patch_embeddings = self.perceiver_resampler(patch_embeddings)  # (b * t, 1, n_patch_latents, patch_feat_dim)
            patch_embeddings = patch_embeddings.squeeze(1)  # (b * t, n_patch_latents, patch_feat_dim)
            patch_embeddings = patch_embeddings.view(batch_size, sequence_length, self.n_patch_latents, self.patch_feat_dim)  # (b, t, n_patch_latents, patch_feat_dim)
            if self.use_hand_rgb:
                hand_patch_embeddings = hand_patch_embeddings.unsqueeze(1)
                hand_patch_embeddings = self.perceiver_resampler(hand_patch_embeddings)
                hand_patch_embeddings = hand_patch_embeddings.squeeze(1)
                hand_patch_embeddings = hand_patch_embeddings.view(batch_size, sequence_length, self.n_patch_latents, self.patch_feat_dim)  # (b, t, n_patch_latents, patch_feat_dim)
            
            # Embed images and patches
            obs_embeddings = self.embed_img(obs_embeddings.float())  # (b, t, h)
            patch_embeddings = self.embed_patch(patch_embeddings.float())  # (b, t, n_patch_latents, h)
            if self.use_hand_rgb:
                hand_obs_embeddings = self.embed_hand_img(hand_obs_embeddings.float())  # (b, t, h)
                hand_patch_embeddings = self.embed_hand_patch(hand_patch_embeddings.float())  # (b, t, n_patch_latents, h)
            
            # Add timestep embeddings
            time_embeddings = self.embed_timestep.weight  # (l, h)
            lang_embeddings = lang_embeddings.view(batch_size, 1, -1) + time_embeddings
            state_embeddings = state_embeddings + time_embeddings
            patch_embeddings = patch_embeddings + time_embeddings.view(sequence_length, 1, self.hidden_size)
            obs_embeddings = obs_embeddings + time_embeddings
            if self.use_hand_rgb:
                hand_obs_embeddings = hand_obs_embeddings + time_embeddings
                hand_patch_embeddings = hand_patch_embeddings + time_embeddings.view(sequence_length, 1, self.hidden_size)

            # Format sequence: lang, state, patch, obs, hand_patch, hand_obs, [ACT], [OBS], [OBS_HAND]
            lang_embeddings = lang_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
            state_embeddings = state_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
            obs_embeddings = obs_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
            stacked_inputs = torch.cat(
                    (lang_embeddings, 
                     state_embeddings, 
                     patch_embeddings, 
                     obs_embeddings), dim=2)  # (b, t, n_tokens, h)
            if self.use_hand_rgb:
                hand_obs_embeddings = hand_obs_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
                stacked_inputs = torch.cat(
                    (stacked_inputs,
                     hand_patch_embeddings, 
                     hand_obs_embeddings), dim=2)  # (b, t, n_tokens, h)

            prediction = {
                'obs_targets': obs_targets,
                'obs_hand_targets': obs_hand_targets,
                'stacked_inputs': stacked_inputs,
            }
            return prediction

        elif mode == 'denoise':

            obs_preds = None
            obs_hand_preds = None
            arm_action_preds = None
            gripper_action_preds = None

            batch_size, sequence_length, _, _ = stacked_inputs.shape

            diffusion_step = self.diffusion_step_encoder(diffusion_step) # (b, h)
            diffusion_step = diffusion_step.view(batch_size, 1, 1, self.hidden_size).repeat(1, sequence_length, 1, 1) # TODO: (b t 1 h)
            stacked_inputs = torch.cat((stacked_inputs, diffusion_step), dim=2) # (b, t, n_tokens, h)

            if self.act_pred:
                action_queries = self.action_queries.weight  # (chunk_size, h)
                action_queries = action_queries.view(1, 1, self.chunk_size, self.hidden_size).repeat(batch_size, sequence_length, 1, 1) # (b, t, chunk_size, h)
                action_queries += self.embed_noisy_actions(noisy_actions)  # (b, t, chunk_size, h)
                stacked_inputs = torch.cat((stacked_inputs, action_queries), dim=2)  # (b, t, n_tokens, h)
            if self.fwd_pred:
                obs_queries = self.obs_queries.weight  # (n_patch_latents + 1, h)
                obs_queries = obs_queries.view(1, 1, self.n_patch_latents + 1, self.hidden_size).repeat(batch_size, sequence_length, 1, 1)  # (b, t, n_patch_latents + 1, h)
                stacked_inputs = torch.cat((stacked_inputs, obs_queries), dim=2)
                if self.fwd_pred_hand:
                    obs_hand_queries = self.obs_hand_queries.weight # 10, h
                    obs_hand_queries = obs_hand_queries.view(1, 1, self.n_patch_latents+1, self.hidden_size).repeat(batch_size, sequence_length, 1, 1)
                    stacked_inputs = torch.cat((stacked_inputs, obs_hand_queries), dim=2)
            
            # Number of tokens
            n_lang_tokens = 1
            n_state_tokens = 1
            n_patch_tokens = self.n_patch_latents
            n_obs_tokens = 1
            n_hand_patch_tokens = self.n_patch_latents
            n_hand_obs_tokens = 1
            n_tokens = n_lang_tokens + n_state_tokens + n_patch_tokens + n_obs_tokens
            if self.use_hand_rgb:
                n_tokens += n_hand_obs_tokens
                n_tokens += n_hand_patch_tokens
            n_diffusion_step_tokens = 1
            n_tokens += 1
            if self.act_pred:
                n_act_pred_tokens = self.chunk_size
                act_query_token_start_i = n_tokens
                n_tokens += self.chunk_size
            if self.fwd_pred:
                obs_query_token_start_i = n_tokens
                n_tokens += (n_patch_tokens + n_obs_tokens)
                if self.fwd_pred_hand:
                    obs_hand_query_token_start_i = n_tokens
                    n_tokens += (n_patch_tokens + n_obs_tokens) 

            # Layer norm
            stacked_inputs = stacked_inputs.reshape(batch_size, n_tokens * sequence_length, self.hidden_size)
            stacked_inputs = self.embed_ln(stacked_inputs)

            # Attention mask
            stacked_attention_mask = attention_mask.view(batch_size, sequence_length, 1)
            if self.use_hand_rgb:
                stacked_attention_mask = stacked_attention_mask.repeat(
                    1, 1, n_lang_tokens + n_state_tokens + n_hand_patch_tokens + n_hand_obs_tokens + n_patch_tokens + n_obs_tokens) # (b, t, n_tokens)
            else:
                stacked_attention_mask = stacked_attention_mask.repeat(
                    1, 1, n_lang_tokens + n_state_tokens + n_patch_tokens + n_obs_tokens)
            if self.act_pred:
                act_query_attention_mask = torch.zeros((batch_size, sequence_length, n_act_pred_tokens+n_diffusion_step_tokens), dtype=torch.long, device=stacked_inputs.device) # not attended to
                stacked_attention_mask = torch.cat((stacked_attention_mask, act_query_attention_mask), dim=2)
            if self.fwd_pred:
                obs_query_attention_mask = torch.zeros((batch_size, sequence_length, n_patch_tokens + n_obs_tokens), dtype=torch.long, device=stacked_inputs.device) # not attended to
                stacked_attention_mask = torch.cat((stacked_attention_mask, obs_query_attention_mask), dim=2)
                if self.fwd_pred_hand:
                    obs_hand_query_attention_mask = torch.zeros((batch_size, sequence_length, n_patch_tokens + n_obs_tokens), dtype=torch.long, device=stacked_inputs.device) # not attened to
                    stacked_attention_mask = torch.cat((stacked_attention_mask, obs_hand_query_attention_mask), dim=2)
            stacked_attention_mask = stacked_attention_mask.reshape(batch_size, n_tokens * sequence_length)

            # GPT forward pass
            transformer_outputs = self.transformer(
                inputs_embeds=stacked_inputs,
                attention_mask=stacked_attention_mask,
            )
            x = transformer_outputs['last_hidden_state']
            x = x.reshape(batch_size, sequence_length, n_tokens, self.hidden_size)

            # Action prediction
            if self.act_pred:
                action_embedding = x[:, :, act_query_token_start_i:(act_query_token_start_i+self.chunk_size)]
                for pred_act_mlp in self.pred_act_mlps:
                    action_embedding = pred_act_mlp(action_embedding)
                arm_action_preds = self.pred_arm_act(action_embedding)  # (b, t, act_dim - 1)
                gripper_action_preds = self.pred_gripper_act(action_embedding)  # (b, t, 1)
                
            # Forward prediction
            if self.fwd_pred:
                mask_token = self.mask_token  # (1, 1, 1, h)
                mask_tokens = mask_token.repeat(batch_size, sequence_length, (self.image_size//self.patch_size)**2, 1)  # (b, t, n_patches, h)
                mask_tokens = mask_tokens + self.decoder_pos_embed.unsqueeze(0).repeat(batch_size, sequence_length, 1, 1)  # (b, t, n_patches, h)

                obs_pred = self.decoder_embed(x[:, :, obs_query_token_start_i:(obs_query_token_start_i + self.n_patch_latents + n_obs_tokens)])  # (b, t, n_patch_latents + 1, h)
                obs_pred_ = torch.cat([obs_pred, mask_tokens + self.embed_noisy_obs(noisy_obs)], dim=2)  # (b, t, n_patches + n_patch_latens + 1, h)
                obs_pred_ = obs_pred_.reshape(-1, obs_pred_.shape[-2], obs_pred_.shape[-1])  # (b * t, n_patches + n_patch_latens + 1, h)
                for blk in self.decoder_blocks:
                    obs_pred_ = blk(obs_pred_)
                obs_pred_ = self.decoder_norm(obs_pred_)
                obs_preds = self.decoder_pred(obs_pred_)  # (b * t, n_patches + n_patch_latens + 1, h)
                obs_preds = obs_preds.reshape(batch_size, sequence_length, -1, obs_preds.shape[-1])  # (b, t, n_patches + n_patch_latens + 1, h)
                obs_preds = obs_preds[:, :, (self.n_patch_latents+n_obs_tokens):]  # (b, t, n_patches, h)

                if self.fwd_pred_hand:
                    obs_pred_hand = self.decoder_embed(x[:, :, obs_hand_query_token_start_i:(obs_hand_query_token_start_i + self.n_patch_latents + n_obs_tokens)])
                    obs_pred_hand_ = torch.cat([obs_pred_hand, mask_tokens + self.embed_noisy_obs(noisy_hand_obs)], dim=2)
                    obs_pred_hand_ = obs_pred_hand_.reshape(-1, obs_pred_hand_.shape[-2], obs_pred_hand_.shape[-1])
                    for blk in self.decoder_blocks:
                        obs_pred_hand_ = blk(obs_pred_hand_)
                    obs_pred_hand_ = self.decoder_norm(obs_pred_hand_)
                    obs_hand_preds = self.decoder_pred(obs_pred_hand_)
                    obs_hand_preds = obs_hand_preds.reshape(batch_size, sequence_length, -1, obs_hand_preds.shape[-1])
                    obs_hand_preds = obs_hand_preds[:, :, (self.n_patch_latents+n_obs_tokens):]
            
            prediction = {
                'obs_preds': obs_preds,
                'obs_hand_preds': obs_hand_preds,
                'arm_action_preds': arm_action_preds,
                'gripper_action_preds': gripper_action_preds,
            }
            return prediction
