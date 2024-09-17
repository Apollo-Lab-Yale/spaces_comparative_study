import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm
import cv2
import pandas as pd
import yaml
import collections
import os

from dp_utils.network import create_nets

from mavis_mujoco_gym.utils.create_env import create_env

import utils

os.environ['MUJOCO_GL'] = 'egl'


env_configs = {
            "render_fps": 50,
            "robot_noise_ratio": 0.01,
            "obs_space_type": "config_space",
            "act_space_type": "config_space",
            "action_normalization": True,
            "observation_normalization": True,
            "render_mode": "rgb_array",
            "img_width": 640,
            "img_height": 480,
            "enable_rgb": True,
            "enable_depth": False,
            "use_cheating_observation": False
        }

env_id = "Microwave-v0"

num_eval_rollouts = 100

normalize_rgb_images = True

img_width = 256
img_height = 192

max_eval_steps = 500

max_num_ckpts = 11

pred_horizon = 16
obs_horizon = 2
action_horizon = 8

vision_feature_dim = 512

num_diffusion_iters = 100

seed = 0

training_session_ckpts_dir = "ckpts/mujoco_microwave_vision/final_config_space_num_demo_rollouts_174"

experiment_result_saving_dir = "experiments/mujoco_microwave_vision/training_session_eval_results/config_space"

if not os.path.exists(experiment_result_saving_dir):
    os.makedirs(experiment_result_saving_dir)

env_config_file_path = os.path.join(experiment_result_saving_dir, "env_configs.yaml")

with open(env_config_file_path, "w") as f:
    yaml.dump(env_configs)

eval_config_file_path = os.path.join(experiment_result_saving_dir, "eval_configs.yaml")

with open(eval_config_file_path, "w") as f:
    yaml.dump({
        "env_id": env_id,
        "num_eval_rollouts": num_eval_rollouts,
        "normalize_rgb_images": normalize_rgb_images,
        "img_width": img_width,
        "img_height": img_height,
        "max_eval_steps": max_eval_steps,
        "pred_horizon": pred_horizon,
        "obs_horizon": obs_horizon,
        "action_horizon": action_horizon,
        "vision_feature_dim": vision_feature_dim,
        "num_diffusion_iters": num_diffusion_iters,
        "seed": seed,
        "training_session_ckpts_dir": training_session_ckpts_dir
    })

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(seed)
np.random.seed(seed)


if env_configs["obs_space_type"] == "config_space":
    lowdim_obs_dim = 17
elif env_configs["obs_space_type"] == "eef_euler_space":
    lowdim_obs_dim = 13
elif env_configs["obs_space_type"] == "eef_quat_space":
    lowdim_obs_dim = 15
elif env_configs["obs_space_type"] == "eef_axis_angle_space":
    lowdim_obs_dim = 15
elif env_configs["obs_space_type"] == "lookat_euler_space":
    lowdim_obs_dim = 10
elif env_configs["obs_space_type"] == "lookat_quat_space":
    lowdim_obs_dim = 11
elif env_configs["obs_space_type"] == "lookat_axis_angle_space":
    lowdim_obs_dim = 11
else:
    raise ValueError("Invalid obs_space_type")

if env_configs["act_space_type"] == "config_space":
    action_dim = 17
elif env_configs["act_space_type"] == "eef_euler_space":
    action_dim = 13
elif env_configs["act_space_type"] == "eef_quat_space":
    action_dim = 15
elif env_configs["act_space_type"] == "eef_axis_angle_space":
    action_dim = 15
elif env_configs["act_space_type"] == "lookat_euler_space":
    action_dim = 10
elif env_configs["act_space_type"] == "lookat_quat_space":
    action_dim = 11
elif env_configs["act_space_type"] == "lookat_axis_angle_space":
    action_dim = 11
else:
    raise ValueError("Invalid act_space_type")

nets = create_nets(
    vision_feature_dim=vision_feature_dim,
    lowdim_obs_dim=lowdim_obs_dim,
    action_dim=action_dim,
    obs_horizon=obs_horizon
)

noise_pred_net = nets["noise_pred_net"]
vision_encoder = nets["vision_encoder"]

noise_pred_net = noise_pred_net.to(device)
vision_encoder = vision_encoder.to(device)

ema = EMAModel(
    parameters=nets.parameters(),
    power=0.75
)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
)

if os.path.exists(training_session_ckpts_dir):
    ckpt_files = os.listdir(training_session_ckpts_dir)
    ckpt_files = [ckpt_file for ckpt_file in ckpt_files if ckpt_file.endswith('.ckpt')]

    if len(ckpt_files) > 0:
        ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split('_')[1]))

        ckpts_epoch_nums = []
        ckpts_mean_cumulative_rewards = []
        ckpts_std_cumulative_rewards = []
        ckpts_mean_rollout_lengths = []
        ckpts_std_rollout_lengths = []
        ckpts_success_rates = []

        num_ckpts = 0

        for ckpt_file in ckpt_files:
            ckpt_file_dir = os.path.join(training_session_ckpts_dir, ckpt_file)

            curr_epoch_num = int(ckpt_file.split('_')[1])

            ckpts_epoch_nums.append(curr_epoch_num)

            curr_epoch_logging_dir = os.path.join(experiment_result_saving_dir, f"epoch_{curr_epoch_num}")

            if not os.path.exists(curr_epoch_logging_dir):
                os.makedirs(curr_epoch_logging_dir)

            video_saving_dir = os.path.join(curr_epoch_logging_dir, "rollout_videos")

            if not os.path.exists(video_saving_dir):
                os.makedirs(video_saving_dir)

            rollout_data_saving_dir = os.path.join(curr_epoch_logging_dir, "rollout_data_logging")

            if not os.path.exists(rollout_data_saving_dir):
                os.makedirs(rollout_data_saving_dir)

            state_dict = torch.load(ckpt_file_dir, map_location=device)

            new_state_dict = {}

            for key, value in state_dict.items():
                new_key = key.replace("module.", "")
                new_state_dict[new_key] = value

            ema_nets = nets

            ema_nets.load_state_dict(new_state_dict)

            rgb_stat = {
                    'min': np.tile(np.array([0.0, 0.0, 0.0]),
                                   (obs_horizon, img_height, img_width, 1)),
                    'max': np.tile(np.array([255.0, 255.0, 255.0]),
                                   (obs_horizon, img_height, img_width, 1))
            }

            rollout_cumulative_rewards = []
            rollout_reward_means = []
            rollout_reward_stds = []
            rollout_lengths = []
            is_rollout_success = []

            env = create_env(env_id, env_configs)

            with tqdm(total=num_eval_rollouts, desc="Number of Evaluation Rollouts") as pbar:
                for rollout_idx in range(num_eval_rollouts):

                    obs, info = env.reset()

                    obs['rgb'] = cv2.resize(obs['rgb'], (img_width, img_height))

                    obs_deque = collections.deque(
                        [obs] * obs_horizon, maxlen=obs_horizon)

                    imgs = [env.render()]
                    rewards = list()
                    done = False
                    eval_step_idx = 0

                    terminated = False
                    truncated = False

                    with tqdm(total=max_eval_steps, desc="Number of Evaluation Steps") as pbar2:
                        while eval_step_idx < max_eval_steps and not done:
                            B = 1

                            images = np.stack([x['rgb'] for x in obs_deque])
                            agent_poses = np.stack([x['state'] for x in obs_deque])

                            nagent_poses = agent_poses

                            nimages = (images - rgb_stat['min']) / (rgb_stat['max'] - rgb_stat['min'])
                            nimages = nimages * 2 - 1

                            nimages = np.transpose(nimages, (0, 3, 1, 2))

                            nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
                            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)

                            with torch.no_grad():
                                image_features = ema_nets['vision_encoder'](nimages)
                                obs_features = torch.cat([image_features, nagent_poses], dim=-1)
                                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                                noisy_action = torch.randn(
                                    (B, pred_horizon, action_dim), device=device)
                                naction = noisy_action

                                noise_scheduler.set_timesteps(num_diffusion_iters)

                                for k in noise_scheduler.timesteps:
                                    noise_pred = ema_nets['noise_pred_net'](
                                        sample=naction,
                                        timestep=k,
                                        global_cond=obs_cond
                                    )

                                    naction = noise_scheduler.step(
                                        model_output=noise_pred,
                                        timestep=k,
                                        sample=naction
                                    ).prev_sample

                                naction = naction.detach().to('cpu').numpy()

                                action_pred = naction[0]

                                start = obs_horizon - 1
                                end = start + action_horizon
                                action = action_pred[start:end, :]

                            for i in range(len(action)):
                                obs, reward, terminated, truncated, info = env.step(action[i])

                                obs['rgb'] = cv2.resize(obs['rgb'], (img_width, img_height))

                                done = terminated or truncated

                                obs_deque.append(obs)
                                rewards.append(reward)

                                imgs.append(env.render())

                                eval_step_idx += 1
                                pbar2.update(1)
                                pbar2.set_postfix(reward=reward)
                                if eval_step_idx > max_eval_steps:
                                    done = True
                                if done:
                                    break

                    env.reset()

                    rollout_cumulative_rewards.append(np.sum(rewards))
                    rollout_reward_means.append(np.mean(rewards))
                    rollout_reward_stds.append(np.std(rewards))
                    rollout_lengths.append(len(rewards))

                    # save rollout data as pandas dataframe
                    rollout_data = {
                        "rewards": rewards,
                    }

                    rollout_data_file_path = os.path.join(rollout_data_saving_dir, f"rollout_{len(rollout_cumulative_rewards) - 1}.csv")
                    df = pd.DataFrame(rollout_data)
                    df.to_csv(rollout_data_file_path)

                    if terminated and not truncated:
                        print("Rollout Success")

                        is_rollout_success.append(True)

                        # Get the number of successful rollouts from is_rollout_success
                        curr_num_of_successful_rollouts = np.sum(is_rollout_success)
                        print("Current Total Number of Successful Rollouts: ", curr_num_of_successful_rollouts)

                    else:
                        is_rollout_success.append(False)

                    imgs = np.stack(imgs)
                    imgs = np.expand_dims(imgs, axis=0)  # Add batch dimension
                    imgs = torch.from_numpy(imgs)
                    imgs = imgs.permute(0, 1, 4, 2, 3)  # Rearrange dimensions to (N, T, C, H, W)

                    video_saving_path = os.path.join(video_saving_dir, f"rollout_{len(rollout_cumulative_rewards) - 1}.mp4")

                    utils.save_video_from_tensor(imgs, video_saving_path)

                    pbar.update(1)

            eval_results = {
                "rollout_cumulative_rewards": rollout_cumulative_rewards,
                "rollout_reward_means": rollout_reward_means,
                "rollout_reward_stds": rollout_reward_stds,
                "rollout_lengths": rollout_lengths,
                "is_rollout_success": is_rollout_success
            }

            eval_results_file_path = os.path.join(curr_epoch_logging_dir, "eval_results.csv")
            df = pd.DataFrame(eval_results)
            df.to_csv(eval_results_file_path)

            mean_cumulative_reward = np.mean(rollout_cumulative_rewards)
            std_cumulative_reward = np.std(rollout_cumulative_rewards)
            mean_rollout_length = np.mean(rollout_lengths)
            std_rollout_length = np.std(rollout_lengths)
            success_rate = np.mean(is_rollout_success)

            env.close()

            ckpts_mean_cumulative_rewards.append(mean_cumulative_reward)
            ckpts_std_cumulative_rewards.append(std_cumulative_reward)
            ckpts_mean_rollout_lengths.append(mean_rollout_length)
            ckpts_std_rollout_lengths.append(std_rollout_length)
            ckpts_success_rates.append(success_rate)

            num_ckpts += 1

            if num_ckpts > max_num_ckpts:
                break

        eval_results_summary = {
            "ckpts_epoch_nums": ckpts_epoch_nums,
            "ckpts_mean_cumulative_rewards": ckpts_mean_cumulative_rewards,
            "ckpts_std_cumulative_rewards": ckpts_std_cumulative_rewards,
            "ckpts_mean_rollout_lengths": ckpts_mean_rollout_lengths,
            "ckpts_std_rollout_lengths": ckpts_std_rollout_lengths,
            "ckpts_success_rates": ckpts_success_rates
        }

        eval_results_summary_file_path = os.path.join(experiment_result_saving_dir, "eval_results_summary.csv")
        df = pd.DataFrame(eval_results_summary)
        df.to_csv(eval_results_summary_file_path)
else:
    print("No checkpoints found in the training session ckpts directory. Exiting...")
