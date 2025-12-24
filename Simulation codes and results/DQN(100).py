# DQN (100)
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.01, gamma=0.95, epsilon=0.1, buffer_size=20000, batch_size=128, target_update_freq=50):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step = 0
        self.memory = deque(maxlen=buffer_size)
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step += 1
        if self.step % self.target_update_freq == 0:
            self.update_target()
grid_size_km = 10
cell_size_km = 0.1
num_4g = 50
num_5g = 50
np.random.seed(42)
grid_centers_per_axis = int(grid_size_km / 1)
centers_x = np.linspace(0.5, grid_size_km - 0.5, grid_centers_per_axis)
centers_y = np.linspace(0.5, grid_size_km - 0.5, grid_centers_per_axis)
centers = np.array(np.meshgrid(centers_x, centers_y)).T.reshape(-1, 2)
indices = np.random.permutation(len(centers))
coords_4g = centers[indices[:num_4g]]
coords_5g = centers[indices[num_4g:num_4g + num_5g]]
def assign_frequency(coords, reuse_factor=7):
    kmeans = KMeans(n_clusters=reuse_factor, random_state=42)
    kmeans.fit(coords)
    freqs = kmeans.labels_ % reuse_factor
    return np.array(freqs)
lte_freqs = assign_frequency(coords_4g)
nr_freqs = assign_frequency(coords_5g)
num_uavs = 4
uav_z = 100
uav_speed_kmh = 100
uav_speed_kms = uav_speed_kmh / 3600
wavelength = 1.0
pass_length_km = 38.202
measurement_interval = 0.1
quadrants = [
    (0, 5, 0, 5),
    (5, 10, 0, 5),
    (0, 5, 5, 10),
    (5, 10, 5, 10)
]
lte_config = {'freq':0.9,'ptx':36,'phi_3db':30,'theta_3db':12,'gmax':18,'am':20, 'sectors':[0,120,240],'height':25,'bw_hz':5e6,'noise':-174 + 10*np.log10(5e6)+3}
nr_config = {'freq':2.1,'ptx':39,'phi_3db':5,'theta_3db':6,'gmax':24,'am':20, 'sectors':[0,120,240],'height':30,'bw_hz':10e6,'noise':-174 + 10*np.log10(10e6)+3}
def norm_angle(angle): return (angle + 180) % 360 - 180
def antenna_gain(phi, theta, phi_b, theta_b, phi_3db, theta_3db, gmax, am):
    phi_diff = norm_angle(phi - phi_b)
    theta_diff = theta - theta_b
    att = np.minimum(12*(phi_diff/phi_3db)**2 + 12*(theta_diff/theta_3db)**2, am)
    return gmax - att
def lte_gain(phi, theta):
    return max(antenna_gain(phi, theta, s, -6, lte_config['phi_3db'], lte_config['theta_3db'], lte_config['gmax'], lte_config['am']) for s in lte_config['sectors'])
def nr_gain(phi, theta):
    return max(max(antenna_gain(phi, theta, s+b, 0, nr_config['phi_3db'], nr_config['theta_3db'], nr_config['gmax'], nr_config['am']) for b in np.linspace(-32.5, 32.5, 24)) for s in nr_config['sectors'])
def tr_36_777_path_loss(d_2d, d_3d, freq, h_uav, h_bs):
    theta = np.degrees(np.arctan2(h_uav-h_bs, d_2d))
    a,b = 10,0.3
    P_LoS = 1/(1+a*np.exp(-b*(theta-a)))
    is_LoS = np.random.rand() < P_LoS
    if is_LoS:
        PL = 28 + 22*np.log10(d_3d) + 20*np.log10(freq)
        sf_std = 4
    else:
        PL_LoS = 28 + 22*np.log10(d_3d) + 20*np.log10(freq)
        PL_NLoS = 32.4 + 23*np.log10(d_3d) + 20*np.log10(freq)
        PL = max(PL_LoS, PL_NLoS)
        sf_std = 6
    SF = np.random.normal(0, sf_std)
    return min(PL,150), SF
def get_sinr(x, y, bs_coords, bs_height, ptx, freq, gain_func, noise_dbm, h_uav, bs_freqs):
    rsrps = []
    freq_indices = []
    for idx, bs in enumerate(bs_coords):
        d_hor = np.hypot(x-bs[0], y-bs[1])*1000
        dz = h_uav - bs_height
        d_3d = np.hypot(d_hor, dz)
        theta = np.degrees(np.arctan2(dz, d_hor))
        phi = np.degrees(np.arctan2(y-bs[1], x-bs[0]))
        gain = gain_func(phi, theta)
        PL, SF = tr_36_777_path_loss(d_hor, d_3d, freq, h_uav, bs_height)
        rx = ptx + gain - PL + SF
        rsrps.append(rx)
        freq_indices.append(bs_freqs[idx])
    rsrps = np.array(rsrps)
    freq_indices = np.array(freq_indices)
    sinrs = []
    interference_powers = []
    for i in range(len(bs_coords)):
        signal = rsrps[i]
        same_freq = freq_indices == freq_indices[i]
        same_freq[i] = False
        interferer_rsrps = rsrps[same_freq]
        interference = max(np.sum(10**(interferer_rsrps/10)),1e-12)
        sinr_lin = 10**(signal/10)/(10**(noise_dbm/10) + interference)
        sinr = 10*np.log10(sinr_lin)
        sinrs.append(sinr)
        interference_powers.append(10*np.log10(interference))
    return rsrps, sinrs, noise_dbm, interference_powers
def eta_from_sinr(sinr_db):
    if sinr_db < -10: return 0.03
    elif sinr_db < -7.5: return 0.06
    elif sinr_db < -5: return 0.10
    elif sinr_db < -2.5: return 0.30
    elif sinr_db < 0: return 0.60
    elif sinr_db < 2.5: return 1.20
    elif sinr_db < 5: return 1.62
    elif sinr_db < 7.5: return 2.00
    elif sinr_db < 10: return 2.64
    elif sinr_db < 12.5: return 3.20
    elif sinr_db < 15: return 3.80
    elif sinr_db < 17.5: return 4.20
    elif sinr_db < 20: return 4.60
    elif sinr_db < 22.5: return 5.00
    elif sinr_db < 25: return 5.30
    elif sinr_db < 30: return 5.55
    elif sinr_db < 35: return 6.00
    elif sinr_db < 40: return 6.30
    else: return 6.57
class DRLHandover:
    def __init__(self, num_lte, num_nr, grid_size_km, cell_size_km, ttt_steps=20, hysteresis_margin=20):
        self.num_lte = num_lte
        self.num_nr = num_nr
        self.grid_size_km = grid_size_km
        self.cell_size_km = cell_size_km
        self.max_x_bin = int(grid_size_km/cell_size_km) - 1
        self.max_y_bin = self.max_x_bin
        self.max_cell = max(num_lte, num_nr)-1
        self.sinr_bins = [-10,-7,-4,-1,2,5,8,12,18,float('inf')]
        self.max_sinr_bin = len(self.sinr_bins)-1
        self.ttt_steps = ttt_steps
        self.hysteresis_margin = hysteresis_margin
        self.dqn_agent = DQNAgent(5,3,lr=0.01,gamma=0.95,epsilon=0.1)
        self.ttt_counter = 0
        self.last_target = None
    def get_state(self, x, y, system, cell_idx, sinr_db):
        x_bin = min(int(x/self.cell_size_km), self.max_x_bin)
        y_bin = min(int(y/self.cell_size_km), self.max_y_bin)
        sinr_bin = next(i for i,b in enumerate(self.sinr_bins) if sinr_db <= b)
        return (x_bin, y_bin, system, cell_idx, sinr_bin)
    def encode_state(self, state_tuple):
        x_bin, y_bin, system, cell_idx, sinr_bin = state_tuple
        system_idx = 0 if system=='LTE' else 1
        return np.array([x_bin/self.max_x_bin, y_bin/self.max_y_bin, system_idx, cell_idx/self.max_cell, sinr_bin/self.max_sinr_bin], dtype=float)
    def choose_action(self, state_tuple, lte_sinrs, nr_sinrs, current_R, lte_bw, nr_bw):
        state = self.encode_state(state_tuple)
        best_lte_sinr = max(lte_sinrs) if len(lte_sinrs)>0 else -float('inf')
        best_nr_sinr = max(nr_sinrs) if len(nr_sinrs)>0 else -float('inf')
        best_lte_R = lte_bw * eta_from_sinr(best_lte_sinr) / 1e6 if len(lte_sinrs) > 0 else -float('inf')
        best_nr_R = nr_bw * eta_from_sinr(best_nr_sinr) / 1e6 if len(nr_sinrs) > 0 else -float('inf')
        target_action = 0
        if best_lte_R > current_R + self.hysteresis_margin:
            target_action = 1
        elif best_nr_R > current_R + self.hysteresis_margin:
            target_action = 2
        if target_action == self.last_target:
            self.ttt_counter += 1
        else:
            self.ttt_counter = 1
            self.last_target = target_action
        if self.ttt_counter < self.ttt_steps:
            target_action = 0
        if random.random() < self.dqn_agent.epsilon:
            return target_action
        else:
            state_tensor = torch.FloatTensor(self.encode_state(state_tuple)).unsqueeze(0)
            with torch.no_grad():
                q_values = self.dqn_agent.model(state_tensor)[0]
            if best_lte_R <= current_R + self.hysteresis_margin:
                q_values[1] = -float('inf')
            if best_nr_R <= current_R + self.hysteresis_margin:
                q_values[2] = -float('inf')
            return q_values.argmax().item()
    def get_next_from_action(self, action, current_system, current_bs_idx, lte_sinrs, nr_sinrs):
        if action == 0:
            return current_system, current_bs_idx
        elif action == 1:
            return 'LTE', np.argmax(lte_sinrs) if len(lte_sinrs)>0 else current_bs_idx
        elif action == 2:
            return 'NR', np.argmax(nr_sinrs) if len(nr_sinrs)>0 else current_bs_idx
    def update(self, state_tuple, action, reward, next_state_tuple):
        state = self.encode_state(state_tuple)
        next_state = self.encode_state(next_state_tuple)
        self.dqn_agent.remember(state, action, reward, next_state, False)
        self.dqn_agent.replay()
    def get_reward(self, R, sinr_db, is_handover):
        if sinr_db < 0:
            reward = -1
        elif 0 <= sinr_db < 10:
            reward = -0.5
        elif 10 <= sinr_db < 20:
            reward = 0.5
        else:
            reward = 1
        if is_handover:
            reward -= 0.2
        return reward
def run_simulation(total_time_s, epsilon, q_agent, lte_config, nr_config, coords_4g, coords_5g, lte_freqs, nr_freqs, uav_z, uav_speed_kms, measurement_interval, quadrants, pass_length_km, num_uavs, save_q_table=True, output_dir="uav_datasets", suffix=""):
    num_steps = int(total_time_s / measurement_interval)
    x_range_km = (total_time_s * uav_speed_kms / pass_length_km) * 5
    all_uav_paths_x = []
    all_uav_paths_y = []
    for i in range(num_uavs):
        x_min, x_max, y_min, y_max = quadrants[i]
        x = np.linspace(0, x_range_km, num_steps)
        y = (y_max + y_min)/2 + ((y_max - y_min)/2)*0.8 * np.sin(2 * np.pi * x / wavelength)
        x_shifted = x + x_min
        all_uav_paths_x.append(x_shifted)
        all_uav_paths_y.append(y)
    q_agent.dqn_agent.epsilon = epsilon
    all_serving_systems = []
    all_serving_cells_idx = []
    all_serving_rsrps = []
    all_handover_times = []
    all_lte_sinrs = []
    all_nr_sinrs = []
    all_lte_rsrps = []
    all_nr_rsrps = []
    all_lte_noise = []
    all_nr_noise = []
    all_lte_interference = []
    all_nr_interference = []
    all_lte_handover_counts = []
    all_nr_handover_counts = []
    all_max_sinrs = []
    all_min_sinrs = []
    all_negative_sinp_counts = []
    start_time = time.time()
    for uav_idx in range(num_uavs):
        uav_path_x = all_uav_paths_x[uav_idx]
        uav_path_y = all_uav_paths_y[uav_idx]
        serving_systems = []
        serving_cells_idx = []
        serving_rsrps = []
        handover_times = []
        lte_sinrs_all = []
        nr_sinrs_all = []
        lte_rsrps_all = []
        nr_rsrps_all = []
        lte_noise_all = []
        nr_noise_all = []
        lte_interference_all = []
        nr_interference_all = []
        lte_handover_count = 0
        nr_handover_count = 0
        lte_rsrps, lte_sinrs, lte_noise_dbm, lte_interference = get_sinr(uav_path_x[0], uav_path_y[0], coords_4g, lte_config['height'], lte_config['ptx'], lte_config['freq'], lte_gain, lte_config['noise'], uav_z, lte_freqs)
        nr_rsrps, nr_sinrs, nr_noise_dbm, nr_interference = get_sinr(uav_path_x[0], uav_path_y[0], coords_5g, nr_config['height'], nr_config['ptx'], nr_config['freq'], nr_gain, nr_config['noise'], uav_z, nr_freqs)
        if len(lte_rsrps) == 0 and len(nr_rsrps) == 0:
            print(f"UAV {uav_idx + 1}: No base stations available at initial position.")
            continue
        max_lte_R = lte_config['bw_hz'] * eta_from_sinr(max(lte_sinrs)) / 1e6 if len(lte_sinrs) > 0 else -float('inf')
        max_nr_R = nr_config['bw_hz'] * eta_from_sinr(max(nr_sinrs)) / 1e6 if len(nr_sinrs) > 0 else -float('inf')
        if max_lte_R >= max_nr_R:
            current_system = 'LTE'
            current_bs_idx = np.argmax(lte_sinrs) if len(lte_sinrs) > 0 else 0
            current_rsrp = lte_rsrps[current_bs_idx]
            current_sinr = lte_sinrs[current_bs_idx]
            current_R = max_lte_R
        else:
            current_system = 'NR'
            current_bs_idx = np.argmax(nr_sinrs)
            current_rsrp = nr_rsrps[current_bs_idx]
            current_sinr = nr_sinrs[current_bs_idx]
            current_R = max_nr_R
        serving_systems.append(current_system)
        serving_cells_idx.append(current_bs_idx)
        serving_rsrps.append(current_rsrp)
        lte_rsrps_all.append(lte_rsrps)
        nr_rsrps_all.append(nr_rsrps)
        lte_sinrs_all.append(lte_sinrs)
        nr_sinrs_all.append(nr_sinrs)
        lte_noise_all.append(lte_noise_dbm)
        nr_noise_all.append(nr_noise_dbm)
        lte_interference_all.append(lte_interference)
        nr_interference_all.append(nr_interference)
        for t in range(1, len(uav_path_x)):
            x, y = uav_path_x[t], uav_path_y[t]
            lte_rsrps, lte_sinrs, lte_noise_dbm, lte_interference = get_sinr(x, y, coords_4g, lte_config['height'], lte_config['ptx'], lte_config['freq'], lte_gain, lte_config['noise'], uav_z, lte_freqs)
            nr_rsrps, nr_sinrs, nr_noise_dbm, nr_interference = get_sinr(x, y, coords_5g, nr_config['height'], nr_config['ptx'], nr_config['freq'], nr_gain, nr_config['noise'], uav_z, nr_freqs)
            current_state_tuple = q_agent.get_state(x, y, current_system, current_bs_idx, current_sinr)
            action = q_agent.choose_action(current_state_tuple, lte_sinrs, nr_sinrs, current_R, lte_config['bw_hz'], nr_config['bw_hz'])
            next_system, next_bs_idx = q_agent.get_next_from_action(action, current_system, current_bs_idx, lte_sinrs, nr_sinrs)
            sinr_db = lte_sinrs[next_bs_idx] if next_system == 'LTE' else nr_sinrs[next_bs_idx]
            R = (lte_config['bw_hz'] if next_system == 'LTE' else nr_config['bw_hz']) * eta_from_sinr(sinr_db) / 1e6
            is_handover = (next_system != current_system) or (next_bs_idx != current_bs_idx)
            reward = q_agent.get_reward(R, sinr_db, is_handover)
            current_system = next_system
            current_bs_idx = next_bs_idx
            current_rsrp = lte_rsrps[current_bs_idx] if current_system == 'LTE' else nr_rsrps[current_bs_idx]
            current_sinr = sinr_db
            current_R = R
            if is_handover:
                handover_times.append(t * measurement_interval)
                if current_system == 'LTE':
                    lte_handover_count += 1
                else:
                    nr_handover_count += 1
            next_state_tuple = q_agent.get_state(x, y, current_system, current_bs_idx, current_sinr)
            q_agent.update(current_state_tuple, action, reward, next_state_tuple)
            serving_systems.append(current_system)
            serving_cells_idx.append(current_bs_idx)
            serving_rsrps.append(current_rsrp)
            lte_rsrps_all.append(lte_rsrps)
            nr_rsrps_all.append(nr_rsrps)
            lte_sinrs_all.append(lte_sinrs)
            nr_sinrs_all.append(nr_sinrs)
            lte_noise_all.append(lte_noise_dbm)
            nr_noise_all.append(nr_noise_dbm)
            lte_interference_all.append(lte_interference)
            nr_interference_all.append(nr_interference)
        serving_sinrs = []
        negative_sinp_count = 0
        for step in range(1, len(serving_systems)):
            system = serving_systems[step]
            cell_idx = serving_cells_idx[step]
            sinr_db = lte_sinrs_all[step - 1][cell_idx] if system == 'LTE' else nr_sinrs_all[step - 1][cell_idx]
            if not np.isnan(sinr_db):
                serving_sinrs.append(sinr_db)
                if sinr_db < 0:
                    negative_sinp_count += 1
        max_sinp = max(serving_sinrs) if serving_sinrs else -float('inf')
        min_sinp = min(serving_sinrs) if serving_sinrs else float('inf')
        all_serving_systems.append(serving_systems)
        all_serving_cells_idx.append(serving_cells_idx)
        all_serving_rsrps.append(serving_rsrps)
        all_handover_times.append(handover_times)
        all_lte_sinrs.append(lte_sinrs_all)
        all_nr_sinrs.append(nr_sinrs_all)
        all_lte_rsrps.append(lte_rsrps_all)
        all_nr_rsrps.append(nr_rsrps_all)
        all_lte_noise.append(lte_noise_all)
        all_nr_noise.append(nr_noise_all)
        all_lte_interference.append(lte_interference_all)
        all_nr_interference.append(nr_interference_all)
        all_lte_handover_counts.append(lte_handover_count)
        all_nr_handover_counts.append(nr_handover_count)
        all_max_sinrs.append(max_sinp)
        all_min_sinrs.append(min_sinp)
        all_negative_sinp_counts.append(negative_sinp_count)
    if save_q_table:
        torch.save(q_agent.dqn_agent.model.state_dict(), f'dqn_model{suffix}.pth')
    E_S_bits, E_Q_bits = 1000, 4000
    L_HO_LTE, L_HO_NR = 0.03, 0.015
    all_throughputs = []
    all_latencies = []
    all_total_handovers = []
    all_max_throughputs = []
    all_min_throughputs = []
    for uav_idx in range(num_uavs):
        throughputs = []
        latencies = []
        serving_systems = all_serving_systems[uav_idx]
        serving_cells_idx = all_serving_cells_idx[uav_idx]
        lte_sinrs_all = all_lte_sinrs[uav_idx]
        nr_sinrs_all = all_nr_sinrs[uav_idx]
        handover_times = all_handover_times[uav_idx]
        for idx, system in enumerate(serving_systems):
            if idx == 0:
                continue
            sinr_db = lte_sinrs_all[idx - 1][serving_cells_idx[idx]] if system == 'LTE' else nr_sinrs_all[idx - 1][serving_cells_idx[idx]]
            eta = eta_from_sinr(sinr_db)
            bw_hz = lte_config['bw_hz'] if system == 'LTE' else nr_config['bw_hz']
            R = bw_hz * eta / 1e6
            throughputs.append(R)
            L_net = (E_S_bits + E_Q_bits) / (R * 1e6)
            is_handover = (idx * measurement_interval) in handover_times
            L_ho_delay = L_HO_LTE if system == 'LTE' else L_HO_NR
            L_total = L_net + (L_ho_delay if is_handover else 0)
            latencies.append(L_total)
        all_throughputs.append(throughputs)
        all_latencies.append(latencies)
        all_total_handovers.append(len(handover_times))
        max_throughput = np.max(throughputs) if throughputs else 0
        min_throughput = np.min(throughputs) if throughputs else 0
        all_max_throughputs.append(max_throughput)
        all_min_throughputs.append(min_throughput)
    os.makedirs(output_dir, exist_ok=True)
    for uav_idx in range(num_uavs):
        dataset_rows = []
        uav_x = all_uav_paths_x[uav_idx]
        uav_y = all_uav_paths_y[uav_idx]
        serving_sys = all_serving_systems[uav_idx]
        serving_idx = all_serving_cells_idx[uav_idx]
        serving_rsrps = all_serving_rsrps[uav_idx]
        lte_rsrps = all_lte_rsrps[uav_idx]
        nr_rsrps = all_nr_rsrps[uav_idx]
        lte_sinrs = all_lte_sinrs[uav_idx]
        nr_sinrs = all_nr_sinrs[uav_idx]
        lte_noise = all_lte_noise[uav_idx]
        nr_noise = all_nr_noise[uav_idx]
        lte_interference = all_lte_interference[uav_idx]
        nr_interference = all_nr_interference[uav_idx]
        handover_times = set(all_handover_times[uav_idx])
        for step in range(1, len(uav_x)):
            time_s = step * measurement_interval
            system = serving_sys[step]
            cell_idx = serving_idx[step]
            rsrp = serving_rsrps[step]
            lte_rsrp_max = max(lte_rsrps[step - 1]) if lte_rsrps[step - 1].size > 0 else float('-inf')
            nr_rsrp_max = max(nr_rsrps[step - 1]) if nr_rsrps[step - 1].size > 0 else float('-inf')
            sinr_db = lte_sinrs[step - 1][cell_idx] if system == 'LTE' else nr_sinrs[step - 1][cell_idx]
            noise_dbm = lte_noise[step - 1] if system == 'LTE' else nr_noise[step - 1]
            interference_dbm = lte_interference[step - 1][cell_idx] if system == 'LTE' else nr_interference[step - 1][cell_idx]
            R = all_throughputs[uav_idx][step - 1] if step - 1 < len(all_throughputs[uav_idx]) else 0
            latency = all_latencies[uav_idx][step - 1] if step - 1 < len(all_latencies[uav_idx]) else 0
            handover_flag = 1 if round(time_s, 2) in [round(ht, 2) for ht in handover_times] else 0
            dataset_rows.append([
                round(time_s, 2), round(uav_x[step], 4), round(uav_y[step], 4),
                system, cell_idx, round(rsrp, 2),
                round(lte_rsrp_max, 2) if lte_rsrp_max != float('-inf') else '',
                round(nr_rsrp_max, 2) if nr_rsrp_max != float('-inf') else '',
                round(noise_dbm, 2), round(interference_dbm, 2) if interference_dbm != -float('inf') else '',
                round(sinr_db, 2) if not np.isnan(sinr_db) else '',
                round(R, 2), round(latency, 4), handover_flag
            ])
        headers = [
            "time_s", "uav_x_km", "uav_y_km",
            "serving_system", "serving_cell_index", "serving_rsrp_dbm",
            "lte_max_rsrp_dbm", "nr_max_rsrp_dbm",
            "noise_power_dbm", "interference_dbm", "sinr_db",
            "throughput_mbps", "latency_s", "handover"
        ]
        file_path = os.path.join(output_dir, f"uav_{uav_idx + 1}_dataset{suffix}.csv")
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(dataset_rows)
    print(f"\nSimulation time for {total_time_s} seconds: {time.time() - start_time:.2f} seconds")
    for uav_idx in range(num_uavs):
        avg_throughput = np.mean(all_throughputs[uav_idx]) if all_throughputs[uav_idx] else 0
        avg_latency = np.mean(all_latencies[uav_idx]) if all_latencies[uav_idx] else 0
        total_handovers = all_total_handovers[uav_idx]
        lte_handovers = all_lte_handover_counts[uav_idx]
        nr_handovers = all_nr_handover_counts[uav_idx]
        max_sinp = all_max_sinrs[uav_idx]
        min_sinp = all_min_sinrs[uav_idx]
        negative_sinp_count = all_negative_sinp_counts[uav_idx]
        max_throughput = all_max_throughputs[uav_idx]
        min_throughput = all_min_throughputs[uav_idx]
        print(f"\nUAV {uav_idx + 1} ({total_time_s}s):")
        print(f"Average AMC-based throughput: {avg_throughput:.2f} Mbps")
        print(f"Highest throughput: {max_throughput:.2f} Mbps")
        print(f"Lowest throughput: {min_throughput:.2f} Mbps")
        print(f"Average system latency: {avg_latency:.3f} s")
        print(f"Total handovers: {total_handovers}")
        print(f"Average handovers per 30-second window: {total_handovers / (total_time_s / 30):.2f}")
        print(f"LTE handovers: {lte_handovers}")
        print(f"5G handovers: {nr_handovers}")
        print(f"Highest SINR: {max_sinp:.2f} dB")
        print(f"Lowest SINR: {min_sinp:.2f} dB")
        print(f"Negative SINR occurrences: {negative_sinp_count} ({negative_sinp_count/len(all_serving_systems[uav_idx][1:])*100:.2f}%)")
    plt.figure(figsize=(16, 12))
    colors = ['red', 'green', 'blue', 'purple']
    for uav_idx in range(num_uavs):
        plt.subplot(3, 4, uav_idx + 1)
        plt.plot(all_uav_paths_x[uav_idx], all_uav_paths_y[uav_idx], color=colors[uav_idx], label=f'UAV {uav_idx + 1} (100 km/h)')
        plt.scatter(coords_4g[:, 0], coords_4g[:, 1], c='black', marker='x', label='4G BS')
        plt.scatter(coords_5g[:, 0], coords_5g[:, 1], c='gray', marker='o', label='5G BS')
        plt.title(f'UAV {uav_idx + 1} Path and Base Stations ({total_time_s}s)')
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
        plt.xlim(0, grid_size_km)
        plt.grid(True)
        plt.legend()
        plt.gca().set_aspect('equal')
    for uav_idx in range(num_uavs):
        plt.subplot(3, 4, uav_idx + 5)
        serving_rsrps = all_serving_rsrps[uav_idx]
        handover_times = all_handover_times[uav_idx]
        plt.plot(np.arange(len(serving_rsrps)) * measurement_interval, serving_rsrps, label=f'UAV {uav_idx + 1} RSRP', color=colors[uav_idx])
        handover_indices = (np.array(handover_times) / measurement_interval).astype(int)
        valid_indices = [i for i in handover_indices if i < len(serving_rsrps)]
        plt.scatter(np.array(handover_times)[:len(valid_indices)], [serving_rsrps[i] for i in valid_indices], c='red', marker='*', label='Handovers')
        plt.title(f'UAV {uav_idx + 1} RSRP Over Time ({total_time_s}s)')
        plt.xlabel('Time (s)')
        plt.ylabel('RSRP (dBm)')
        plt.grid(True)
        plt.legend()
    for uav_idx in range(num_uavs):
        plt.subplot(3, 4, uav_idx + 9)
        serving_systems = all_serving_systems[uav_idx]
        serving_cells_idx = all_serving_cells_idx[uav_idx]
        sinrs = [all_lte_sinrs[uav_idx][i-1][serving_cells_idx[i]] if serving_systems[i] == 'LTE' else all_nr_sinrs[uav_idx][i-1][serving_cells_idx[i]] for i in range(1, len(serving_systems))]
        plt.plot(np.arange(len(sinrs)) * measurement_interval, sinrs, label=f'UAV {uav_idx + 1} SINR', color=colors[uav_idx])
        plt.axhline(0, color='black', linestyle='--', label='SINR = 0 dB')
        plt.axhline(-5, color='red', linestyle='--', label='SINR = -5 dB')
        plt.title(f'UAV {uav_idx + 1} SINR Over Time ({total_time_s}s)')
        plt.xlabel('Time (s)')
        plt.ylabel('SINR (dB)')
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.show()
    return all_serving_systems, all_serving_cells_idx, all_serving_rsrps, all_handover_times, all_lte_sinrs, all_nr_sinrs, all_lte_rsrps, all_nr_rsrps, all_lte_noise, all_nr_noise, all_lte_interference, all_nr_interference, all_lte_handover_counts, all_nr_handover_counts, all_max_sinrs, all_min_sinrs, all_negative_sinp_counts, all_throughputs, all_latencies, all_total_handovers, all_max_throughputs, all_min_throughputs
q_agent = DRLHandover(num_4g, num_5g, grid_size_km, cell_size_km)
print("Running first training pass (60 seconds, epsilon=0.1)...")
run_simulation(60, 0.1, q_agent, lte_config, nr_config, coords_4g, coords_5g, lte_freqs, nr_freqs, uav_z, uav_speed_kms, measurement_interval, quadrants, pass_length_km, num_uavs, save_q_table=True, output_dir="DRL_model_dataset_train1", suffix="_train1")
print("Running second training pass (60 seconds, epsilon=0.05)...")
state_dict = torch.load('dqn_model_train1.pth')
q_agent.dqn_agent.model.load_state_dict(state_dict)
q_agent.dqn_agent.update_target()
run_simulation(60, 0.05, q_agent, lte_config, nr_config, coords_4g, coords_5g, lte_freqs, nr_freqs, uav_z, uav_speed_kms, measurement_interval, quadrants, pass_length_km, num_uavs, save_q_table=True, output_dir="DRL_model_dataset_train2", suffix="_train2")
print("Running third training pass (60 seconds, epsilon=0.01)...")
state_dict = torch.load('dqn_model_train2.pth')
q_agent.dqn_agent.model.load_state_dict(state_dict)
q_agent.dqn_agent.update_target()
run_simulation(60, 0.01, q_agent, lte_config, nr_config, coords_4g, coords_5g, lte_freqs, nr_freqs, uav_z, uav_speed_kms, measurement_interval, quadrants, pass_length_km, num_uavs, save_q_table=True, output_dir="DRL_model_dataset_train3", suffix="_train3")
print("Running final 60-second simulation...")
state_dict = torch.load('dqn_model_train3.pth')
q_agent.dqn_agent.model.load_state_dict(state_dict)
q_agent.dqn_agent.update_target()
results_60 = run_simulation(60, 0.01, q_agent, lte_config, nr_config, coords_4g, coords_5g, lte_freqs, nr_freqs, uav_z, uav_speed_kms, measurement_interval, quadrants, pass_length_km, num_uavs, save_q_table=True, output_dir="DRL_model_dataset_60s_new one", suffix="_60s")
print("Running 1376-second simulation...")
state_dict = torch.load('dqn_model_60s.pth')
q_agent.dqn_agent.model.load_state_dict(state_dict)
q_agent.dqn_agent.update_target()
results_1376 = run_simulation(1376, 0.01, q_agent, lte_config, nr_config, coords_4g, coords_5g, lte_freqs, nr_freqs, uav_z, uav_speed_kms, measurement_interval, quadrants, pass_length_km, num_uavs, save_q_table=True, output_dir="Qtable_XGB_100km/h", suffix="_1376s")
import time
start_train = time.time()
model.fit(X_train, y_train)
end_train = time.time()
training_time = end_train - start_train
print(f"Training time: {training_time:.2f} seconds")
start_infer = time.time()
y_pred_proba = model.predict_proba(X_test)[:, 1]
end_infer = time.time()
inference_time = end_infer - start_infer
print(f"Inference time on {X_test.shape[0]} samples: {inference_time:.4f} seconds")
print(f"Average inference time per sample: {inference_time/X_test.shape[0]:.6f} seconds")