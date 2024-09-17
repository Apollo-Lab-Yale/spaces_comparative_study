import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import zarr
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import shutil
from moviepy.editor import ImageSequenceClip
import mavis_mujoco_gym.utils.mavis_utils as mavis_utils
# from sb3_contrib import TQC
# from stable_baselines3.common.buffers import ReplayBuffer
# from stable_baselines3.common.env_util import make_vec_env
# import mavis_mujoco_gym
# from mavis_mujoco_gym.envs.mavis_base.mavis_base_env import MAVISBaseEnv
# from mavis_mujoco_gym.envs.pick_and_place.pick_and_place import PickAndPlace

class ConfigLoss(nn.Module):
    def __init__(self, l2_weight=1.0, angular_weight=1.0):
        super(ConfigLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l2_weight = l2_weight
        self.angular_weight = angular_weight

    def forward(self, predicted, target):
        # Ensure the shape is (batch_size, horizon, configuration_length)
        assert predicted.shape == target.shape
        batch_size, horizon, configuration_length = predicted.shape
        assert configuration_length == 17

        # L2 loss for elements 0, 8, and 16
        l2_indices = [0, 8, 16]
        l2_loss = 0
        for idx in l2_indices:
            l2_loss += self.mse_loss(predicted[:, :, idx], target[:, :, idx])

        # Angular distance for other elements
        def angular_distance(pred, true):
            return torch.min(torch.abs(pred - true), 2 * torch.pi - torch.abs(pred - true))

        angular_loss = 0
        for i in range(configuration_length):
            if i not in l2_indices:
                angular_loss += angular_distance(predicted[:, :, i], target[:, :, i]).mean()

        # Combine losses with weights
        total_loss = self.l2_weight * l2_loss + self.angular_weight * angular_loss
        return total_loss


def save_video_from_tensor(tensor, output_path, fps=30):
    # Convert the PyTorch tensor to a NumPy array
    video_array = tensor.numpy()

    # Rearrange dimensions to (N, T, H, W, C) and swap color channels to RGB
    video_array = np.transpose(video_array, (0, 1, 3, 4, 2))

    # Remove the Batch dimension
    video_array = video_array.squeeze(0)

    # Create a list of frames
    frames = [video_array[i] for i in range(video_array.shape[0])]

    # Create a video clip using MoviePy
    clip = ImageSequenceClip(frames, fps=fps)

    # Write the video clip to an MP4 file
    clip.write_videofile(output_path, codec='libx264')

"""
ENV_DICT = {
    "MAVISBase-v0": MAVISBaseEnv,
    "PickAndPlace-v0": PickAndPlace,
}

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_0, c_0):
        # if it is not batch, add a batch dimension
        is_batch = len(x.shape) == 3
        if not is_batch:
            x = x.unsqueeze(0)
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        out = self.fc(out.data)
        return out, (h_n, c_n)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda(),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda())


class RolloutDataset(Dataset):
    def __init__(self, root_dir, state_key="lookat_space_euler_state_obs", action_key="lookat_space_euler_action"):
        self.root_dir = root_dir
        self.state_key = state_key
        self.action_key = action_key
        self.rollouts = self._load_rollouts()

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        rollout = self.rollouts[idx]
        action = rollout[self.action_key]
        state_obs = rollout[self.state_key]
        seq_length = len(rollout['step_idx'])

        return {
            'action': torch.from_numpy(action),
            'state_obs': torch.from_numpy(state_obs),
            'seq_length': seq_length
        }

    def _load_rollouts(self):
        rollouts = []
        for subdir in os.listdir(self.root_dir):
            rollout_dir = os.path.join(self.root_dir, subdir)
            if os.path.isdir(rollout_dir):
                rollout = {}
                for key in [self.action_key, self.state_key, 'step_idx']:
                    dataset = zarr.open(os.path.join(rollout_dir, key), mode='r')
                    rollout[key] = dataset[:]
                rollouts.append(rollout)
        return rollouts


    def add_rollout(self, rollout_dir):
        rollout = {}
        for key in [self.action_key, self.state_key, 'step_idx']:
            dataset = zarr.open(os.path.join(rollout_dir, key), mode='r')
            rollout[key] = dataset[:]
        self.rollouts.append(rollout)


class RolloutDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        collated_batch = {}
        seq_lengths = [sample['seq_length'] for sample in batch]

        for key in ['action', 'state_obs']:
            sequences = [sample[key] for sample in batch]
            padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
            collated_batch[key] = padded_sequences

        collated_batch['seq_lengths'] = torch.tensor(seq_lengths)
        return collated_batch

    def add_rollout(self, rollout_dir):
        self.dataset.add_rollout(rollout_dir)


def evaluate_model(model, env_id, env_configs, num_episodes=10, max_steps=500, render=False):
    model.eval()
    env_cls = ENV_DICT[env_id]
    env = env_cls(**env_configs)
    device = next(model.parameters()).device

    total_rewards = []

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        h_0, c_0 = model.init_hidden(1)  # Initialize hidden state for a single environment

        for step in range(max_steps):
            if render:
                env.render()

            # print("shape of state: ", state.shape)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                action, (h_0, c_0) = model(state_tensor, h_0, c_0)

            action = action.squeeze().cpu().numpy()
            next_state, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            state = next_state

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)

    env.close()

    cmulative_reward = sum(total_rewards)
    return cmulative_reward


def load_state_action_demonstrations(demo_root_dir, state_key="lookat_space_euler_state_obs", action_key="lookat_space_euler_action"):
    rollout_dirs = sorted([d for d in os.listdir(demo_root_dir) if d.isdigit()])

    step_idx_list = []
    state_obs_list = []
    action_list = []

    for rollout_dir in rollout_dirs:
        step_idx_file = zarr.open(os.path.join(demo_root_dir, rollout_dir, "step_idx"))
        state_obs_file = zarr.open(os.path.join(demo_root_dir, rollout_dir, state_key))
        action_file = zarr.open(os.path.join(demo_root_dir, rollout_dir, action_key))

        step_idx_list.append(step_idx_file[:])
        state_obs_list.append(state_obs_file[:])
        action_list.append(action_file[:])

    step_idx = np.concatenate(step_idx_list, axis=0)
    state_obs = np.concatenate(state_obs_list, axis=0)
    actions = np.concatenate(action_list, axis=0)

    return step_idx, state_obs, actions

def load_demonstration_to_tqc(demo_root_dir, buffer_size, env_id, env_configs, hyperparams):
    rollout_dirs = sorted([d for d in os.listdir(demo_root_dir) if d.isdigit()])

    env = make_vec_env(env_id, n_envs=1, env_kwargs=env_configs)

    # Create a dummy SAC model to get the ReplayBuffer object
    model = TQC("MlpPolicy",
                env,
                learning_rate=hyperparams["learning_rate"],
                buffer_size=hyperparams["buffer_size"],
                batch_size=hyperparams["batch_size"],
                tau=hyperparams["tau"],
                gamma=hyperparams["gamma"],
                ent_coef=hyperparams["ent_coef"],
                use_sde=hyperparams["use_sde"],
                sde_sample_freq=hyperparams["sde_sample_freq"],
                verbose=1,
                gradient_steps=hyperparams["gradient_steps"],
                policy_kwargs=hyperparams[
                    "policy_kwargs"])

    total_samples = 0
    for rollout_dir in rollout_dirs:
        state_obs_file = zarr.open(os.path.join(demo_root_dir, rollout_dir, "lookat_space_euler_state_obs"))
        action_file = zarr.open(os.path.join(demo_root_dir, rollout_dir, "lookat_space_euler_action"))
        reward_file = zarr.open(os.path.join(demo_root_dir, rollout_dir, "step_rewards"))

        num_samples = len(state_obs_file)

        for i in range(num_samples):
            state_obs = state_obs_file[i]
            action = action_file[i]
            reward = reward_file[i]
            next_state_obs = state_obs_file[i + 1] if i < num_samples - 1 else state_obs_file[i]
            done = i == num_samples - 1
            info = [{} for _ in range(env.num_envs)]

            model.replay_buffer.add(state_obs, next_state_obs, action, reward, done, info)
            total_samples += 1

            if total_samples >= buffer_size:
                break

        if total_samples >= buffer_size:
            break

    return model


def duplicate_files(source_dir, target_dir):
    # Check if the destination directory exists, if not, create it
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Loop through all files in the source directory
    for item in os.listdir(source_dir):
        source_path = os.path.join(source_dir, item)
        destination_path = os.path.join(target_dir, item)

        # If it's a file, copy it
        if os.path.isfile(source_path):
            shutil.copy2(source_path, destination_path)
        # If it's a directory, copy it recursively
        elif os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path)
            
"""
