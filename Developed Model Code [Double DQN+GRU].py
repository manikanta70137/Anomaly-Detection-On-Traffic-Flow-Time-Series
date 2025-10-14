import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import time
import sklearn.preprocessing
import warnings
from datetime import datetime
os.environ["OMP_NUM_THREADS"] = "8"  # Reduced for RAM headroom
os.environ["MKL_NUM_THREADS"] = "8"
import gc

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


class DQNAgent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, input_steps, hidden_dims,
                 n_layers, mem_size, batch_size, eps_min=0.01, eps_dec=5e-6,
                 replace=500, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.input_steps = input_steps
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.action_space = list(range(self.n_actions))
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_steps, input_dims)
        self.q_eval = DeepQNetwork(lr, input_steps, input_dims, n_actions, hidden_dims, n_layers, 'q_eval', chkpt_dir)
        self.q_next = DeepQNetwork(lr, input_steps, input_dims, n_actions, hidden_dims, n_layers, 'q_next', chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            obs_array = np.array(observation, dtype=np.float32)
            state = T.tensor(obs_array, device=self.q_eval.device).unsqueeze(0)
            with T.no_grad():
                actions = self.q_eval(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(states, dtype=T.float32, device=self.q_eval.device)
        actions = T.tensor(actions, dtype=T.int64, device=self.q_eval.device)
        rewards = T.tensor(rewards, dtype=T.float32, device=self.q_eval.device)
        states_ = T.tensor(states_, dtype=T.float32, device=self.q_eval.device)
        dones = T.tensor(dones, dtype=T.bool, device=self.q_eval.device)

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        indices = np.arange(self.batch_size)
        q_pred = self.q_eval(states)[indices, actions]

        with T.no_grad():
            next_actions = self.q_eval(states_).argmax(dim=1)
            q_next = self.q_next(states_).gather(1, next_actions.unsqueeze(1)).squeeze()
            q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next
        loss = T.nn.functional.smooth_l1_loss(q_pred, q_target)
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.q_eval.parameters(), max_norm=1.0)
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_steps, input_dims, n_actions, hidden_dims, n_layers, name, chkpt_dir):
        super().__init__()
        self.input_steps = input_steps
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.n_actions = n_actions
        self.n_layers = n_layers
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name)

        self.gru = nn.GRU(input_dims, hidden_dims, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dims, n_actions)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x, _ = self.gru(x)
        return self.fc(x[:, -1, :])

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ReplayBuffer:
    def __init__(self, max_size, input_steps, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((max_size, input_steps, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((max_size, input_steps, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(max_size, dtype=np.int64)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        return (self.state_memory[batch], self.action_memory[batch],
                self.reward_memory[batch], self.new_state_memory[batch],
                self.terminal_memory[batch])


NOT_ANOMALY = 0
ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]
IDX_MAX = 480
N_STEPS = 20


class EnvTafficRepo:
    def __init__(self, repodir, file_raw, file_avg, file_reward):
        self.repodir = repodir
        self.timeseries_raw = pd.read_csv(os.path.join(repodir, file_raw), usecols=[0, 1], names=['idx', 'value']).values.astype(np.float32)
        self.timeseries_avg = pd.read_csv(os.path.join(repodir, file_avg), usecols=[0, 1], names=['idx', 'value']).values.astype(np.float32)
        self.cluster_reward = pd.read_csv(os.path.join(repodir, file_reward), header=None).values.astype(np.float32)
        self.avg_lookup = dict(zip(self.timeseries_avg[:, 0], self.timeseries_avg[:, 1]))
        self.timeseries_curser = N_STEPS
        self.detected_result = np.zeros((27840, 4), dtype=np.float32)  # idx, value, action, his_mean
        self.result_counter = 0
        self._normalize_data()

    def _normalize_data(self):
        scaler = sklearn.preprocessing.MinMaxScaler()
        self.timeseries_raw[:, 1] = scaler.fit_transform(self.timeseries_raw[:, 1].reshape(-1, 1)).flatten()
        self.timeseries_avg[:, 1] = scaler.fit_transform(self.timeseries_avg[:, 1].reshape(-1, 1)).flatten()

    def reset(self):
        self.timeseries_curser = N_STEPS
        self.result_counter = 0
        initial_state = np.zeros((N_STEPS, 3), dtype=np.float32)
        for i in range(N_STEPS):
            initial_state[i] = [
                self.timeseries_raw[i, 1],
                self.avg_lookup[self.timeseries_raw[i, 0]],
                NOT_ANOMALY
            ]
        return initial_state

    def step(self, action):
        idx = self.result_counter % 27840  # Circular buffer index
        time_slot = self.timeseries_raw[self.timeseries_curser, 0]
        his_mean = self.avg_lookup[time_slot]

        self.detected_result[idx] = [
            self.timeseries_raw[self.timeseries_curser, 0],  # idx
            self.timeseries_raw[self.timeseries_curser, 1],  # value
            action,  # action
            his_mean  # his_mean
        ]
        self.result_counter += 1

        reward = self.cluster_reward[self.timeseries_curser, action]
        self.timeseries_curser += 1
        done = self.timeseries_curser >= len(self.timeseries_raw)
        next_state = self._get_next_state()

        return next_state, reward, done, {}

    def _get_next_state(self):
        valid_entries = min(self.result_counter, 27840)
        current_data = self.detected_result[:valid_entries, 1:4]  # Get available data

        if current_data.shape[0] < N_STEPS:
            padding = np.zeros((N_STEPS - current_data.shape[0], 3), dtype=np.float32)
            current_data = np.concatenate([padding, current_data])
        else:
            current_data = current_data[-N_STEPS:]

        new_state = current_data[-N_STEPS:].copy()
        new_state = np.roll(new_state, -1, axis=0)

        if self.timeseries_curser < len(self.timeseries_raw):
            time_slot = self.timeseries_raw[self.timeseries_curser, 0]
            his_mean = self.avg_lookup[time_slot]
            new_state[-1] = [
                self.timeseries_raw[self.timeseries_curser, 1],
                his_mean,
                -1  # Placeholder for next action
            ]
        return new_state.astype(np.float32)


# Modified config section
DATA_PATH = 'traffic_data'
FILE_NAME = 'real_world_data'
GAMMA = 0.99
EPSILON = 0.3
LEARNING_RATE = 0.0001
NUM_LAYERS = 1  # Reduced from 2
INPUT_DIM = 3
INPUT_SEQ_LENGTH = 20
HIDDEN_DIM = 64  # Reduced from 128
BUFFER_SIZE = 27840  # Reduced from 10000
BATCH_SIZE = 16  # Reduced from 32
NUM_EPOCH = 8


def TrafficAD(with_ground_truth=False):
    env = EnvTafficRepo(DATA_PATH, f'{FILE_NAME}.csv', f'{FILE_NAME}_mean.csv', f'{FILE_NAME}_reward.csv')
    agent = DQNAgent(GAMMA, EPSILON, LEARNING_RATE, 2, INPUT_DIM, INPUT_SEQ_LENGTH,
                     HIDDEN_DIM, NUM_LAYERS, BUFFER_SIZE, BATCH_SIZE)

    total_start = time.time()
    print(f"\nTraining started at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Device: {agent.q_eval.device}")

    for epoch in range(NUM_EPOCH):
        epoch_start = time.time()
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state

            total_reward += reward
            step_count += 1

            # Print progress every 50 steps
            if step_count % 50 == 0:
                print(f"Epoch {epoch + 1} | Step {step_count:4d} | "
                      f"Current Reward: {reward:7.2f} | Avg Reward: {total_reward / step_count:7.2f}")

        epoch_time = time.time() - epoch_start
        avg_reward = total_reward / step_count if step_count > 0 else 0

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCH} completed in {epoch_time:.1f} seconds")
        print(f"Steps: {step_count} | Total Reward: {total_reward:.2f} | "
              f"Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}\n")

    total_time = time.time() - total_start
    print(f"\nTraining completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")

    # Ensure the directory exists and save the results
    os.makedirs(DATA_PATH, exist_ok=True)  # Create 'traffic_data' directory if it doesn't exist
    result_file = f"{DATA_PATH}/{FILE_NAME}_results.csv"
    pd.DataFrame(
        env.detected_result[:env.result_counter],  # Only save valid entries up to result_counter
        columns=['idx', 'value', 'action', 'his_mean']
    ).to_csv(result_file, index=False)
    print(f"Results saved to {result_file}")

    return env.detected_result


if __name__ == '__main__':
    TrafficAD(with_ground_truth=False)

