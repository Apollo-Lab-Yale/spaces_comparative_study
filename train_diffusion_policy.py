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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter
import yaml
import collections
import os
from dp_utils.datasets import DiffusionPolicyVisionDataset
import utils
from dp_utils.unet import ConditionalUnet1D
import socket

from dp_utils.network import create_nets

from mavis_mujoco_gym.utils.create_env import create_env

os.environ['MUJOCO_GL'] = 'egl'

use_eval = False

env_configs = {
            "render_fps": 50,
            "robot_noise_ratio": 0.01,
            "obs_space_type": "eef_euler_space",
            "act_space_type": "eef_euler_space",
            "action_normalization": True,
            "observation_normalization": True,
            "render_mode": "rgb_array",
            "img_width": 640,
            "img_height": 480,
            "enable_rgb": True,
            "enable_depth": False,
            "use_cheating_observation": False
        }


dataset_dir = "demonstrations/realworld_drawer_interaction_occluded_new_conversion_2"
state_key = "eef_space_euler_state_obs"
action_key = "eef_space_euler_action"
normalize_rgb_images = True

training_session_name = "realworld_drawer_interaction_occluded_new_conversion_2/eef_euler_67"

env_id = "Microwave-v0"

batch_size = 64
num_workers = 16
num_epochs = 601
log_interval = 25
max_eval_steps = 500

pred_horizon = 16
obs_horizon = 2
action_horizon = 8

vision_feature_dim = 512

num_warmup_steps = 1000
num_diffusion_iters = 100

seed = 0

log_dir = f"logs/{training_session_name}"
ckpt_dir = f"ckpts/{training_session_name}"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

with open(ckpt_dir + '/hyperparams.yaml', 'w') as f:
    yaml.dump({
        "env_id": env_id,
        "state_key": state_key,
        "action_key": action_key,
        "normalize_rgb_images": normalize_rgb_images,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "num_epochs": num_epochs,
        "log_interval": log_interval,
        "max_eval_steps": max_eval_steps,
        "pred_horizon": pred_horizon,
        "obs_horizon": obs_horizon,
        "action_horizon": action_horizon,
        "vision_feature_dim": vision_feature_dim,
        "num_warmup_steps": num_warmup_steps,
        "num_diffusion_iters": num_diffusion_iters,
        "seed": seed
    })

with open(ckpt_dir + '/env_configs.yaml', 'w') as f:
    yaml.dump(env_configs)

################################################################################

if __name__ == "__main__":
    curr_step = 0
    start_epoch_idx = 0

    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    torch.manual_seed(seed)
    np.random.seed(seed)



    writer = SummaryWriter(log_dir)

    dataset = DiffusionPolicyVisionDataset(
        dataset_path=dataset_dir,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        state_key=state_key,
        action_key=action_key,
        normalize_rgb_images=normalize_rgb_images)

    sampled_rgb_images, sampled_state, sampled_action, sampled_reward = dataset[0]

    # print("sampled_rgb_images shape: ", sampled_rgb_images.shape)
    img_channels, img_height, img_width = sampled_rgb_images.shape[1:]

    lowdim_obs_dim = sampled_state.shape[1]
    action_dim = sampled_action.shape[1]

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            sampler=DistributedSampler(dataset),
                            pin_memory=True,
                            persistent_workers=True)

    if os.path.exists(ckpt_dir):
        ckpt_files = os.listdir(ckpt_dir)
        ckpt_files = [ckpt_file for ckpt_file in ckpt_files if ckpt_file.endswith('.ckpt')]
        if len(ckpt_files) > 0:
            ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_ckpt_file = ckpt_files[-1]
            latest_ckpt_file = os.path.join(ckpt_dir, latest_ckpt_file)
            state_dict = torch.load(latest_ckpt_file, map_location='cuda')
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("module.", "")
                new_state_dict[new_key] = value
            nets = create_nets(
                vision_feature_dim=vision_feature_dim,
                lowdim_obs_dim=lowdim_obs_dim,
                action_dim=action_dim,
                obs_horizon=obs_horizon
            )
            nets.load_state_dict(new_state_dict)
            print("latest_ckpt_file: ", latest_ckpt_file)
            latest_ckpt_file_name = latest_ckpt_file.split("/")[-1]
            parts = latest_ckpt_file_name.split("_")
            start_epoch_idx = int(parts[1])
            curr_step = int(parts[3].split(".")[0])
            print('Loaded checkpoint: ', latest_ckpt_file)
            print('Starting from epoch: ', start_epoch_idx)
            print('Starting from step: ', curr_step)

        else:
            nets = create_nets(
                vision_feature_dim=vision_feature_dim,
                lowdim_obs_dim=lowdim_obs_dim,
                action_dim=action_dim,
                obs_horizon=obs_horizon
            )
    else:
        nets = create_nets(
            vision_feature_dim=vision_feature_dim,
            lowdim_obs_dim=lowdim_obs_dim,
            action_dim=action_dim,
            obs_horizon=obs_horizon
        )

    noise_pred_net = nets['noise_pred_net']
    vision_encoder = nets['vision_encoder']



    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    nets['noise_pred_net'] = DDP(nets['noise_pred_net'].to(device), device_ids=[local_rank])
    nets['vision_encoder'] = DDP(nets['vision_encoder'].to(device), device_ids=[local_rank])

    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4 * dist.get_world_size(),
        weight_decay=1e-6)

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(dataloader) * num_epochs
    )

    with tqdm(range(start_epoch_idx, num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            dataloader.sampler.set_epoch(epoch_idx)
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # for item in nbatch:
                    #     print(item.shape)

                    # data normalized in dataset
                    # device transfer
                    nimage = nbatch[0][:, :obs_horizon].to(device)

                    # print("shape of nimage during training: ", nimage.shape)

                    nagent_pos = nbatch[1][:, :obs_horizon].to(device)
                    naction = nbatch[2].to(device)

                    # print("shape of naction during training: ", naction.shape)

                    B = nagent_pos.shape[0]
                    # Print B with text B:

                    # encoder vision features
                    image_features = nets['vision_encoder'](
                        nimage.flatten(end_dim=1).float())
                    image_features = image_features.reshape(
                        *nimage.shape[:2], -1)
                    # (B,obs_horizon,D)
                    # Print image_features and nagent_pos with text
                    # print('Shape of Image Features: ', image_features.shape)
                    # print('Shape of nagent_pos: ', nagent_pos.shape)
                    # concatenate vision feature and low-dim obs
                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    # print('Shape of OBSCOND before: ', obs_features.shape)
                    obs_cond = obs_features.flatten(start_dim=1)

                    # print("shape of obs_cond during training: ", obs_cond.shape)

                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions.float(), timesteps.float(), global_cond=obs_cond.float())

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    dist.barrier()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

                    curr_step += 1
                    writer.add_scalar('Training/Step Loss', loss_cpu, curr_step)

                    # if curr_step % save_step_thresh == 0:
                    #    torch.save(nets.state_dict(), ckpt_saving_dir + '/net_' + str(epoch_idx) + '_step_' + str(curr_step) + '.ckpt')

                    #    temp_ema_nets = nets
                    #    ema.copy_to(temp_ema_nets.parameters())
                    #    torch.save(temp_ema_nets.state_dict(), ckpt_saving_dir + '/ema_net_epoch' + str(epoch_idx) + '_step_' + str(curr_step) + '.ckpt')

            tglobal.set_postfix(loss=np.mean(epoch_loss))
            writer.add_scalar('Training/Epoch Loss', np.mean(epoch_loss), epoch_idx)

            if dist.get_rank() == 0:
                if epoch_idx % log_interval == 0:
                    ema_noise_pred_net = nets
                    ema.copy_to(ema_noise_pred_net.parameters())
                    torch.save(ema_noise_pred_net.state_dict(),
                               ckpt_dir + '/net_' + str(epoch_idx) + '_step_' + str(curr_step) + '.ckpt')
                    if use_eval:
                        env = create_env(env_id, env_configs)

                        obs, info = env.reset()

                        obs['rgb'] = cv2.resize(obs['rgb'], (img_width, img_height))

                        obs_deque = collections.deque(
                                [obs] * obs_horizon, maxlen=obs_horizon)

                        imgs = [env.render()]
                        rewards = list()
                        done = False
                        eval_step_idx = 0

                        with tqdm(total=max_eval_steps, desc="Evaluation") as pbar:
                            while eval_step_idx < max_eval_steps and not done:
                                B = 1
                                # stack the last obs_horizon number of observations
                                images = np.stack([x['rgb'] for x in obs_deque])
                                agent_poses = np.stack([x['state'] for x in obs_deque])

                                nagent_poses = agent_poses

                                rgb_stat = {
                                    'min': np.tile(np.array([0.0, 0.0, 0.0]),
                                                   (obs_horizon, img_height, img_width, 1)),
                                    'max': np.tile(np.array([255.0, 255.0, 255.0]),
                                                   (obs_horizon, img_height, img_width, 1))
                                }



                                nimages = (images - rgb_stat['min']) / (rgb_stat['max'] - rgb_stat['min'])
                                # Scale to [-1, 1]
                                nimages = nimages * 2 - 1

                                nimages = np.transpose(nimages, (0, 3, 1, 2))

                                # device transfer
                                nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
                                nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)

                                # print("shape of nimages during evaluation: ", nimages.shape)
                                # print("shape of nagent_poses during evaluation: ", nagent_poses.shape)

                                with torch.no_grad():
                                    # print("Executing with torch.no_grad()")
                                    # get image features
                                    image_features = ema_noise_pred_net['vision_encoder'](nimages)
                                    # (2,512)

                                    # concat with low-dim observations
                                    obs_features = torch.cat([image_features, nagent_poses], dim=-1)

                                    # reshape observation to (B,obs_horizon*obs_dim)
                                    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                                    # print("shape of obs_cond during evaluation: ", obs_cond.shape)

                                    # initialize action from Guassian noise
                                    noisy_action = torch.randn(
                                        (B, pred_horizon, action_dim), device=device)
                                    naction = noisy_action

                                    # init scheduler
                                    noise_scheduler.set_timesteps(num_diffusion_iters)

                                    for k in noise_scheduler.timesteps:
                                        # predict noise
                                        noise_pred = ema_noise_pred_net['noise_pred_net'](
                                            sample=naction,
                                            timestep=k,
                                            global_cond=obs_cond
                                        )

                                        # inverse diffusion step (remove noise)
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
                                    # stepping env
                                    obs, reward, terminated, truncated, info = env.step(action[i])

                                    obs['rgb'] = cv2.resize(obs['rgb'], (img_width, img_height))

                                    done = terminated or truncated

                                    # save observations
                                    obs_deque.append(obs)
                                    # and reward/vis
                                    rewards.append(reward)

                                    imgs.append(env.render())

                                    # update progress bar
                                    eval_step_idx += 1
                                    pbar.update(1)
                                    pbar.set_postfix(reward=reward)
                                    if eval_step_idx > max_eval_steps:
                                        done = True
                                    if done:
                                        break

                        del env

                        cumulative_reward = sum(rewards)
                        writer.add_scalar('Evaluation/Cumulative Reward', cumulative_reward, epoch_idx)

                        imgs = np.stack(imgs)
                        imgs = np.expand_dims(imgs, axis=0)  # Add batch dimension
                        imgs = torch.from_numpy(imgs)
                        imgs = imgs.permute(0, 1, 4, 2, 3)  # Rearrange dimensions to (N, T, C, H, W)

                        # Reduce the resolution by 4 times for each timestep
                        resized_imgs = []
                        for i in range(imgs.shape[1]):
                            resized_img = F.interpolate(imgs[:, i], scale_factor=0.4, mode='bilinear', align_corners=False)
                            resized_imgs.append(resized_img)
                        imgs = torch.stack(resized_imgs, dim=1)

                        # print("Shape of img tensor after resolution reduction: ", imgs.shape)
                        writer.add_video('Evaluation/Video', imgs, epoch_idx)
    # Weights of the EMA model
    # is used for inference
    ema_nets = nets
    ema.copy_to(ema_nets.parameters())
