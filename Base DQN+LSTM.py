import numpy as np
import random
import torch as T
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from sklearn import metrics
import time
import sklearn.preprocessing
from sklearn.neighbors import KernelDensity
from datetime import datetime

class DQNAgent(object):
    '''
    the implementation of Deep Q Learning
    '''
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, input_steps, hidden_dims,
                 n_layers, mem_size, batch_size, eps_min=0.01, eps_dec=5e-6,
                 replace=500, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
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
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_steps, input_dims, n_actions)

        self.q_eval = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, hidden_dims=self.hidden_dims,
                                    input_dims=self.input_dims, input_steps=self.input_steps, n_layers = self.n_layers,
                                    name='q_eval', chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, hidden_dims=self.hidden_dims,
                                    input_dims=self.input_dims, input_steps=self.input_steps, n_layers = self.n_layers,
                                    name='q_next', chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        r = np.random.random()
        print(r,self.epsilon)
        if r > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        print(action)
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        T.autograd.set_detect_anomaly(True)
        loss.backward(retain_graph=True)
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def update_eps(self, epsilon):
        self.epsilon = epsilon

class ReplayBuffer(object):
    '''
    Replay buffer to store the most current max_size of tuples (state, action, reward, next_state)
    '''
    def __init__(self, max_size, input_steps, input_dims, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_steps, input_dims,),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_steps, input_dims),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int64)

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

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class DeepQNetwork(nn.Module):
    '''
    The DeepQNetwork is built by a two-layer LSTM and a linear layer;
    it takes as input a state which formed by a N_STEPS of three-dimentional vectors and output an action 0/1
    '''
    def __init__(self, lr, input_steps, input_dims, n_actions, hidden_dims, n_layers, name, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.input_steps = input_steps
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.n_actions = n_actions
        self.n_layers = n_layers
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, name)

        self.lstm = nn.LSTM(self.input_dims, self.hidden_dims, self.n_layers, batch_first=True)
        self.ln = nn.Linear(self.hidden_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        print(self.device)
        self.to(device=self.device)

    def forward(self, state):
        lstm_out, self.hidden_cell = self.lstm(state)
        actions = self.ln(lstm_out.index_select(1, T.tensor([1]).to(self.device)).squeeze(1))

        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

NOT_ANOMALY = 0
ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]
IDX_MAX = 480
N_STEPS = 20



def RewardFucClu(clu_rewards, timeseries_curser, action):
    '''
    :param clu_rewards: the pre-computed reward series
    :param timeseries_curser: the current time step
    :param action: the current action
    :return: the current reward
    '''
    return clu_rewards[timeseries_curser][action]




def StateFuc(timeseries, average, timeseries_curser, action, previous_state=[]):
    '''
    :param timeseries: the traffic flow time series
    :param average: the historical mean of each time step
    :param timeseries_curser: current time step
    :param action: the action taken based on the previous state
    :param previous_state: the previous state
    :return: the current state formed by N_STEPS of [flow, historical_mean_flow, action]. Note that the action for the
    current time step is set to be '-1'.
    '''

    # initialize the first state; we assume the first N_STEPS of flow values are NOT_ANOMALY
    if timeseries_curser == N_STEPS:
        state = []
        for i in range(timeseries_curser):
            his_mean = average['value'][timeseries['idx'][i] - 1]
            state.append([timeseries['value'][i], his_mean, NOT_ANOMALY])

        state.pop(0)
        his_mean = average['value'][timeseries['idx'][timeseries_curser] - 1]
        state.append([timeseries['value'][timeseries_curser], his_mean, -1])

        return np.array(state, dtype='float32')

    if timeseries_curser > N_STEPS:
        # copy the previous state
        # replace the action (with value '-1') with actual action taken in the previous time step
        his_mean = average['value'][timeseries['idx'][timeseries_curser] - 1]
        his_mean_pre = average['value'][timeseries['idx'][timeseries_curser - 1] - 1]
        state0 = np.concatenate((previous_state[1:N_STEPS-1],
                                 [[timeseries['value'][timeseries_curser-1], his_mean_pre, action]]))
        state = np.concatenate((state0,
                                [[timeseries['value'][timeseries_curser], his_mean, -1]]))

        return np.array(state, dtype='float32')



class EnvTafficRepo():
    # init the class instance
    def __init__(self, repodir, file_raw, file_avg, file_reward):

        self.repodir = repodir
        self.file_raw = file_raw
        self.file_avg = file_avg
        # the file with pre-computed reward
        self.file_reward = os.path.join(self.repodir, file_reward)
        # the file with the raw traffic flow
        self.timeseries_raw_file = os.path.join(self.repodir, self.file_raw)
        # the file with the pre-computed historical mean
        self.timeseries_avg_file = os.path.join(self.repodir, self.file_avg)

        self.action_space_n = len(action_space)
        self.timeseries_raw = []
        self.timeseries_curser = -1
        self.timeseries_curser_init = N_STEPS
        self.timeseries_states = []
        self.timeseries_avg = []
        self.detected_result = pd.DataFrame(columns=['idx','value','action'])
        self.cluster_reward = []


        self.statefnc = StateFuc
        self.rewardfnc = RewardFucClu

        ts = pd.read_csv(self.timeseries_raw_file, usecols=[0,1], header=None, names=['idx','value'])
        avg = pd.read_csv(self.timeseries_avg_file, usecols=[0,1], header=None, names=['idx','value'])
        self.cluster_reward = pd.read_csv(self.file_reward, usecols=[0,1], header=None).to_numpy()


        ts['value'] = ts['value'].astype(np.float32)
        avg['value'] = avg['value'].astype(np.float32)

        scaler = sklearn.preprocessing.MinMaxScaler()
        scaler.fit(np.array(ts['value']).reshape(-1, 1))
        scaler.fit(np.array(avg['value']).reshape(-1, 1))
        ts['value'] = scaler.transform(np.array(ts['value']).reshape(-1, 1))
        avg['value'] = scaler.transform(np.array(avg['value']).reshape(-1, 1))

        self.timeseries_raw = ts
        self.timeseries_avg = avg
        self.datasetrng = len(self.timeseries_raw)

    # reset the instance
    def reset(self):
        self.timeseries_curser = self.timeseries_curser_init
        self.detected_result = pd.DataFrame(columns=['idx', 'value', 'action'])

        # return the first state, containing the first element of the time series
        self.timeseries_states = self.statefnc(self.timeseries_raw, self.timeseries_avg, self.timeseries_curser, 0)

        # store the first N_STEPS of results
        for i in range(self.timeseries_curser):
            list = pd.Series([self.timeseries_raw['idx'][i], self.timeseries_raw['value'][i], 0],
                             index=self.detected_result.columns)
            self.detected_result = self.detected_result.append(list, ignore_index=True)
        return self.timeseries_states


    # take a step and gain a reward
    def step(self, action):
        # 0. append the result for the current time step
        list = pd.Series([self.timeseries_raw['idx'][self.timeseries_curser],
                          self.timeseries_raw['value'][self.timeseries_curser], action],
                         index=self.detected_result.columns)
        self.detected_result = self.detected_result.append(list, ignore_index=True)

        # 1. get the reward of the action
        reward = self.rewardfnc(self.cluster_reward,
                                self.timeseries_curser, action)

        # 2. get the next state and the done flag after the action
        self.timeseries_curser += 1

        if self.timeseries_curser >= self.timeseries_raw['value'].size:
            done = 1
            state = self.timeseries_states
        else:
            done = 0
            state = self.statefnc(self.timeseries_raw, self.timeseries_avg, self.timeseries_curser, action, self.timeseries_states)

        self.timeseries_states = state

        return state, reward, done, []


    def get_detected_result(self):
        '''
        :return: detected_result: in the format of[[idx_0, value_1, action_1], [idx_2, value_2, action_2], ...]
        '''
        return self.detected_result

def reward_calculation_kde_clustering(input_file, output_file, neighbor_time_step, daily_time_step):
    '''
    :param input_file: the input raw traffic flow data in the format of ['daily_time_idx', 'flow']
    :param output_file: the output reward file in the format of
    :param neighbor_time_step:
    :param daily_time_step:
    :return:
    '''
    # read the flow time series data and normalization
    ts = pd.read_csv(input_file, usecols=[0, 1], header=None, names=['idx', 'value'])
    ts['value'] = ts['value'].astype(np.float32)

    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(np.array(ts['value']).reshape(-1, 1))
    ts['value'] = scaler.transform(np.array(ts['value']).reshape(-1, 1))

    # the dictionary for key:index (start from 0), value:flow
    index_value_dict = {}
    for i in range(len(ts)):
        idx = ts['idx'][i] - 1
        val = ts['value'][i]
        if idx in index_value_dict.keys():
            index_value_dict[idx].append([i, val])
        else:
            index_value_dict[idx] = [[i, val]]

    reward = [None] * len(ts)

    # begin the main loop to calculate the reward
    for i in range(daily_time_step):
        # the flow data at time step i
        pivot = index_value_dict[i]
        # the data to do clustering
        data = np.array(index_value_dict[i])
        for j in range(neighbor_time_step):
            idx = (i - (j + 1)) % daily_time_step
            data = np.concatenate((data, index_value_dict[idx]), axis=0)
        for j in range(neighbor_time_step):
            idx = (i + j + 1) % daily_time_step
            data = np.concatenate((data, index_value_dict[idx]), axis=0)
        # perform clustering on data set

        X = np.array(data)[:, 1].reshape(-1, 1)
        s = np.linspace(0, X.max(), 1000).reshape(-1, 1)
        e = sklearn_kde(X, s)
        from scipy.signal import argrelextrema
        mi, ma = argrelextrema(np.exp(e), np.less)[0], argrelextrema(np.exp(e), np.greater)[0]
        labels = [-1] * len(data)
        if len(s[mi]) == 0:
            for k in range(len(pivot)):
                idx = pivot[k][0]
                ratio = random.choice([-1, 1])
                reward[idx] = [ratio, -ratio]
        else:
            for j in range(len(X)):
                for idx_mi in range(len(s[mi])):
                    if idx_mi == 0:
                        if X[j] >= 0 and X[j] < s[mi][idx_mi]:
                            labels[j] = idx_mi
                            break
                    else:
                        if X[j] >= s[mi][idx_mi-1] and X[j] < s[mi][idx_mi]:
                            labels[j] = idx_mi
                            break
                if X[j] >= s[mi][len(s[mi])-1]:
                    labels[j] = len(s[mi])

            # calculate the number of element in each cluster
            label_num_dict = {}
            for k in range(len(data)):
                if labels[k] in label_num_dict.keys():
                    label_num_dict[labels[k]] += 1
                else:
                    label_num_dict[labels[k]] = 1

            avg_num_clu = len(data) / (len(s[mi]) + 1)

            # calculate the reward value for each data point, each reward in the format [x,y]
            # x: the reward for a=0
            # y: the reward for a=1
            for k in range(len(pivot)):
                lab = labels[k]
                delta = label_num_dict[lab] / avg_num_clu
                idx = pivot[k][0]
                if delta > 1:
                    # in the case this flow in a larger cluster
                    reward[idx] = [delta, -delta]
                else:
                    # in the case this flow in a smaller cluster
                    ratio = 1 / delta
                    reward[idx] = [-ratio, ratio]
    reward_file = open(output_file, '+w')
    for i in range(len(reward)):
        reward_file.write(str(round(reward[i][0], 3)) + ',' + str(round(reward[i][1], 3)) + '\n')
    reward_file.close()


def sklearn_kde(data, points):

    # Silverman bandwidth estimator
    n, d = data.shape
    bandwidth = (n * (d + 2) / 4.)**(-1. / (d + 4))
    # standardize data so that we can use uniform bandwidth
    mu, sigma = np.mean(data, axis=0), np.std(data, axis=0)
    data, points = (data - mu)/sigma, (points - mu)/sigma

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth, rtol=1e-6, atol=1e-6)
    kde.fit(data)
    log_pdf = kde.score_samples(points)

    return np.exp(log_pdf)

def mean_calculation(from_file, to_file, time_slot):
    # read the flow time series data and normalization
    ts = pd.read_csv(from_file, usecols=[0, 1], header=None, names=['idx', 'value'])
    ts['value'] = ts['value'].astype(np.float32)


    # the dictionary for key:index (start from 0), value:flow
    index_value_dict = {}
    for i in range(len(ts)):
        idx = ts['idx'][i]
        val = ts['value'][i]
        if idx in index_value_dict.keys():
            index_value_dict[idx].append(val)
        else:
            index_value_dict[idx] = [val]

    mean_file = open(to_file, "+w")
    for i in range(time_slot):
        idx = i+1
        flow_values = index_value_dict[idx]
        flow_values = np.array(flow_values)
        mean = flow_values.mean()
        mean_file.write(str(int(idx)) + ',' + str(round(mean, 3)) + '\n')
    mean_file.close()

# config
DATA_PATH = 'C:/Users/manik/Desktop/Fprj/traffic_data'
FILE_NAME = 'real_world_data'
GAMMA = 0.99
EPSILON = 0.5
LEARNING_RATE = 0.0001
NUM_LAYERS = 2
INPUT_DIM = 3
INPUT_SEQ_LENGTH = 20
HIDDEN_DIM = 128
BUFFER_SIZE = 10000
EPS_MIN = 0.01
EPS_DEC = 5e-6
BATCH_SIZE = 32
NUM_EPOCH = 8


def post_processing_smoothing(results, k):
    forward = results['action']
    backward = results['action']
    for i in range(len(results)-2*k):
        zeros = 0
        ones = 0
        for j in range(2*k+1):
            if results['action'][i+j] == 0:
                zeros += 1
            else:
                ones += 1
        if zeros > k:
            forward[i+k] = 0
        else:
            forward[i+k] = 1
    for i in range(len(results)-2*k):
        zeros = 0
        ones = 0
        for j in range(2 * k + 1):
            if results['action'][len(results) - 1 - (i + j)] == 0:
                zeros += 1
            else:
                ones += 1
        if zeros > k:
            backward[len(results) - 1 - (i + k)] = 0
        else:
            backward[len(results) - 1 - (i + k)] = 1
    for i in range(len(results)):
        if forward[i] + backward[i] > 0:
            results['action'][i] = 1
        else:
            results['action'][i] = 0
    return results


def performance_evaluation(gound_truth, results):
    gt = []
    pred = []
    for i in range(len(gound_truth)):
        gt.append(gound_truth['anomaly'][i])
        pred.append(results['action'][i])
    acc = metrics.accuracy_score(gt, pred)
    precision, recall, F1, _ = metrics.precision_recall_fscore_support(gt, pred, average='binary')
    print(metrics.confusion_matrix(gt, pred))

    print('acc:', acc, 'precision:', precision, 'recall:', recall, 'F1 score:', F1)
    return 0


def TrafficAD(with_ground_truth):

    env = EnvTafficRepo(DATA_PATH, f'{FILE_NAME}.csv',
                               f'{FILE_NAME}_mean.csv', f'{FILE_NAME}_reward.csv')

    agent = DQNAgent(gamma=GAMMA, epsilon=EPSILON, lr=LEARNING_RATE, n_layers=NUM_LAYERS,
                     input_dims=INPUT_DIM, input_steps=INPUT_SEQ_LENGTH, hidden_dims=HIDDEN_DIM,
                     n_actions=env.action_space_n, mem_size=BUFFER_SIZE, eps_min=EPS_MIN,
                     batch_size=BATCH_SIZE, eps_dec=EPS_DEC)

    load_checkpoint = False
    if load_checkpoint:
        agent.load_models()

    n_steps = 0
    scores, eps_history, steps_array = [], [], []
    final_result = pd.DataFrame(columns=['idx','value','action'])
    start_time = time.time()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    for i in range(NUM_EPOCH):
        done = False
        observation = env.reset()

        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_

            if n_steps % 10000 == 0:
                print(n_steps, score)
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)
        eps_history.append(agent.epsilon)
        current_results = env.get_detected_result()

        # if the detected results for all epochs need to be stored, modify the follow line to append the current_results
        final_result = current_results

    model_save_path = "Model/dqn_model.pth"
    T.save(agent.q_eval.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    end_time = time.time()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)
    print("--- The time for anomaly detection = %s seconds ---" % (end_time - start_time))

    final_result = post_processing_smoothing(final_result, 10)
    final_result.to_csv(f'{DATA_PATH}/{FILE_NAME}_results.csv')



    if not with_ground_truth:
        SYS_FILE_NAME = 'synthetic_data'
        ground_truth = pd.read_csv(f'{DATA_PATH}/{SYS_FILE_NAME}.csv', header=None, usecols=[0, 1, 2], names=
        ['idx', 'value', 'anomaly'])
        # performance evaluation
        performance_evaluation(ground_truth, final_result)

    return final_result

if __name__ == '__main__':
    # generate mean flow data and reward file
    reward_calculation_kde_clustering(f'{DATA_PATH}/{FILE_NAME}.csv', f'{DATA_PATH}/{FILE_NAME}_reward.csv', 2, 480)
    mean_calculation(f'{DATA_PATH}/{FILE_NAME}.csv', f'{DATA_PATH}/{FILE_NAME}_mean.csv', 480)
    with_ground_truth = False
    TrafficAD(with_ground_truth)





