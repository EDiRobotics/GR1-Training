# MIT License

# Copyright (c) 2021 Oier Mees
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Code to evaluate Calvin."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import copy
from moviepy.editor import ImageSequenceClip

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from evaluation.calvin_evaluation import GR1CalvinEvaluation
from utils.calvin_utils import print_and_save

logger = logging.getLogger(__name__)

# Path to calvin
CALVIN_ROOT = os.environ['CALVIN_ROOT']

def make_env(dataset_path, observation_space, device):
    val_folder = Path(dataset_path) / "validation"
    from evaluation.calvin_env_wrapper_raw import CalvinEnvWrapperRaw
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env

def evaluate_policy(model, env, eval_sr_path, eval_result_path, num_sequences, ep_len, eval_dir=None, debug=False):
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    eval_dir = get_log_dir(eval_dir)
    eval_sequences = get_sequences(num_sequences)

    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    sequence_i = 0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, ep_len, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i)
        results.append(result)
        if not debug:
            success_list = count_success(results)
            with open(eval_sr_path, 'a') as f:
                line =f"{sequence_i}/{num_sequences}: "
                for sr in success_list:
                    line += f"{sr:.3f} | "
                sequence_i += 1
                line += "\n"
                f.write(line)
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|"
            )
        else:
            sequence_i += 1
    print_and_save(results, eval_sequences, eval_result_path, None)
    return results


def evaluate_sequence(env, model, task_checker, ep_len, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(env, model, task_checker, ep_len, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, ep_len, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i):
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()
    if debug:
        img_list = []
    for step in range(ep_len):
        action = model.step(obs, lang_annotation)
        obs, _, _, current_info = env.step(action)
        if debug:
            img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
            img_list.append(img_copy)
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
                clip = ImageSequenceClip(img_list, fps=30)
                clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-succ.gif'), fps=30)
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
        clip = ImageSequenceClip(img_list, fps=30)
        clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-fail.gif'), fps=30)
    return False
