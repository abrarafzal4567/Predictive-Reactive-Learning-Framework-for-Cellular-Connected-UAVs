# Proposed approach with XGB model
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from sklearn.cluster import KMeans
import pickle
import time
from collections import defaultdict
import joblib
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
# --- Grid and UAV Setup ---
grid_size_km = 10
cell_size_km = 0.1 # Coarser grid for efficiency
x_coords = np.arange(0, grid_size_km, cell_size_km)
y_coords = np.arange(0, grid_size_km, cell_size_km)
xx, yy = np.meshgrid(x_coords, y_coords)
# Base Station Deployment
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
# Assign Frequency Channels
def assign_frequency(coords, reuse_factor=7):
    kmeans = KMeans(n_clusters=reuse_factor, random_state=42)
    kmeans.fit(coords)
    freqs = kmeans.labels_ % reuse_factor
    return np.array(freqs)
lte_freqs = assign_frequency(coords_4g)
nr_freqs = assign_frequency(coords_5g)
# UAVs Setup
num_uavs = 4
uav_z = 100 # UAV height in meters
uav_speed_kmh = 100
uav_speed_kms = uav_speed_kmh / 3600 # km/s
wavelength = 1.0 # km
pass_length_km = 38.202
measurement_interval = 0.1
# Quadrant boundaries
quadrants = [
    (0, 5, 0, 5),
    (5, 10, 0, 5),
    (0, 5, 5, 10),
    (5, 10, 5, 10)
]
# Antenna Configs
lte_config = {
    'freq': 0.9,
    'ptx': 36,
    'phi_3db': 30,
    'theta_3db': 12,
    'gmax': 18,
    'am': 20,
    'sectors': [0, 120, 240],
    'height': 25,
    'bw_hz': 5e6,
    'noise': -174 + 10 * np.log10(5e6) + 3
}
nr_config = {
    'freq': 2.1,
    'ptx': 39,
    'phi_3db': 5,
    'theta_3db': 6,
    'gmax': 24,
    'am': 20,
    'sectors': [0, 120, 240],
    'height': 30,
    'bw_hz': 10e6,
    'noise': -174 + 10 * np.log10(10e6) + 3
}
# Gain and Propagation Functions
def norm_angle(angle): return (angle + 180) % 360 - 180
def antenna_gain(phi, theta, phi_b, theta_b, phi_3db, theta_3db, gmax, am):
    phi_diff = norm_angle(phi - phi_b)
    theta_diff = theta - theta_b
    att = np.minimum(12 * (phi_diff / phi_3db)**2 + 12 * (theta_diff / theta_3db)**2, am)
    return gmax - att
def lte_gain(phi, theta):
    return max(antenna_gain(phi, theta, s, -6, lte_config['phi_3db'], lte_config['theta_3db'], lte_config['gmax'], lte_config['am']) for s in lte_config['sectors'])
def nr_gain(phi, theta):
    return max(max(antenna_gain(phi, theta, s + b, 0, nr_config['phi_3db'], nr_config['theta_3db'], nr_config['gmax'], nr_config['am'])
                   for b in np.linspace(-32.5, 32.5, 24)) for s in nr_config['sectors'])
def tr_36_777_path_loss(d_2d, d_3d, freq, h_uav, h_bs):
    theta = np.degrees(np.arctan2(h_uav - h_bs, d_2d))
    a, b = 10, 0.3
    P_LoS = 1 / (1 + a * np.exp(-b * (theta - a)))
    is_LoS = np.random.rand() < P_LoS
    if is_LoS:
        PL = 28.0 + 22 * np.log10(d_3d) + 20 * np.log10(freq)
        sf_std = 4
    else:
        PL_LoS = 28.0 + 22 * np.log10(d_3d) + 20 * np.log10(freq)
        PL_NLoS = 32.4 + 23 * np.log10(d_3d) + 20 * np.log10(freq)
        PL = max(PL_LoS, PL_NLoS)
        sf_std = 6
    SF = np.random.normal(0, sf_std)
    PL = min(PL, 150) # Cap path loss to prevent extreme signal loss
    return PL, SF
# SINR Calculation
def get_sinr(x, y, bs_coords, bs_height, ptx, freq, gain_func, noise_dbm, h_uav, bs_freqs):
    rsrps = []
    freq_indices = []
    for idx, bs in enumerate(bs_coords):
        d_hor = np.hypot(x - bs[0], y - bs[1]) * 1000
        dz = h_uav - bs_height
        d_3d = np.hypot(d_hor, dz)
        theta = np.degrees(np.arctan2(dz, d_hor))
        phi = np.degrees(np.arctan2(y - bs[1], x - bs[0]))
        gain = gain_func(phi, theta)
        PL, SF = tr_36_777_path_loss(d_hor, d_3d, freq, h_uav, bs_height)
        ptx_adj = ptx if idx == 0 else ptx - 20
        rx = ptx_adj + gain - PL + SF
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
        valid_inters = interferer_rsrps > -90
        interference = np.sum(10 ** (interferer_rsrps[valid_inters] / 10)) if np.any(valid_inters) else 0
        interference_dbm = 10 * np.log10(interference) if interference > 0 else -float('inf')
        sinr = 10 * np.log10(10 ** (signal / 10) / (10 ** (noise_dbm / 10) + interference))
        sinr = max(sinr)
        sinrs.append(sinr)
        interference_powers.append(interference_dbm)
    return rsrps, sinrs, noise_dbm, interference_powers
# AMC-based Throughput Calculation
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
# Q-Learning Handover Logic with XGBoost
class QLearningHandover:
    def __init__(self, num_lte, num_nr, grid_size_km, cell_size_km):
        self.q_table = defaultdict(float)
        self.alpha = 0.01 # Updated learning rate
        self.gamma = 0.95 # Discount factor
        self.epsilon = 0.1 # Initial exploration rate
        self.num_lte = num_lte
        self.num_nr = num_nr
        self.grid_size_km = grid_size_km
        self.cell_size_km = cell_size_km
        self.sinr_bins = [-10, -5, 0, 5, 10, 15, float('inf')]
        self.rsrp_bins = [-120, -100, -80, -60, float('inf')] # Added for state
        # Load XGBoost model
        self.xgb_model = joblib.load('xgb_handover_model_uav.joblib')
        # Initialize scaler for XGBoost features
        self.scaler = StandardScaler()
        self.theta = 0.8 # Handover probability threshold 
        self.delta = 3.0 # RSRP threshold for candidate cells (dB)
    def fit_scaler(self, X):
        """Fit the scaler with training data for XGBoost features."""
        self.scaler.fit(X)
    def get_state(self, x, y, system, cell_idx, sinr_db, rsrp_db):
        x_bin = int(x / self.cell_size_km)
        y_bin = int(y / self.cell_size_km)
        x_bin = min(x_bin, int(self.grid_size_km / self.cell_size_km) - 1)
        y_bin = min(y_bin, int(self.grid_size_km / self.cell_size_km) - 1)
        sinr_bin = next(i for i, b in enumerate(self.sinr_bins) if sinr_db <= b)
        rsrp_bin = next(i for i, b in enumerate(self.rsrp_bins) if rsrp_db <= b)
        return (x_bin, y_bin, system, cell_idx, sinr_bin, rsrp_bin)
    def get_handover_probability(self, feature_vector):
        selected_features = feature_vector[[0, 1, 4]]
        selected_features_scaled = self.scaler.transform(selected_features.reshape(1, -1))
        prob = self.xgb_model.predict_proba(selected_features_scaled)[:, 1][0]
        return prob
    def get_actions(self, lte_rsrps, nr_rsrps, current_system, current_idx):
        actions = [('stay', current_idx)] # Include stay action
        current_rsrp = lte_rsrps[current_idx] if current_system == 'LTE' else nr_rsrps[current_idx]
        lte_candidates = [(i, p) for i, p in enumerate(lte_rsrps) if (i != current_idx or current_system != 'LTE') and p >= current_rsrp - self.delta]
        nr_candidates = [(i, p) for i, p in enumerate(nr_rsrps) if (i != current_idx or current_system != 'NR') and p >= current_rsrp - self.delta]
        lte_candidates = sorted(lte_candidates, key=lambda x: x[1], reverse=True)[:1]
        nr_candidates = sorted(nr_candidates, key=lambda x: x[1], reverse=True)[:1]
        actions.extend([('LTE', i) for i, _ in lte_candidates])
        actions.extend([('NR', i) for i, _ in nr_candidates])
        return actions
    def get_reward(self, sinr_db, is_handover):
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
    def choose_action(self, state, lte_rsrps, nr_rsrps, current_system, current_idx, feature_vector):
        prob = self.get_handover_probability(feature_vector)
        actions = self.get_actions(lte_rsrps, nr_rsrps, current_system, current_idx)
        # Restrict to stay action if handover probability is below threshold
        if prob < self.theta:
            return ('stay', current_idx)
        if np.random.rand() < self.epsilon:
            return actions[np.random.randint(len(actions))]
        q_values = [self.q_table[(state, action)] for action in actions]
        return actions[np.argmax(q_values)]
    def update_q_table(self, state, action, reward, next_state, actions):
        current_q = self.q_table[(state, action)]
        next_q = [self.q_table[(next_state, a)] for a in actions]
        max_next_q = max(next_q) if next_q else 0
        self.q_table[(state, action)] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
# Training Function
def run_simulation(total_time_s, epsilon, q_agent, lte_config, nr_config, coords_4g, coords_5g, lte_freqs, nr_freqs, uav_z, uav_speed_kms, measurement_interval, quadrants, pass_length_km, num_uavs, save_q_table=True, output_dir="uav_datasets", suffix=""):
    simulation_start = time.time() # Start timing for this simulation run
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
    # Fit scaler with initial feature data
    initial_features = []
    for uav_idx in range(num_uavs):
        uav_path_x = all_uav_paths_x[uav_idx]
        uav_path_y = all_uav_paths_y[uav_idx]
        lte_rsrps, lte_sinrs, _, _ = get_sinr(uav_path_x[0], uav_path_y[0], coords_4g, lte_config['height'], lte_config['ptx'], lte_config['freq'], lte_gain, lte_config['noise'], uav_z, lte_freqs)
        nr_rsrps, nr_sinrs, _, _ = get_sinr(uav_path_x[0], uav_path_y[0], coords_5g, nr_config['height'], nr_config['ptx'], nr_config['freq'], nr_gain, nr_config['noise'], uav_z, nr_freqs)
        if len(lte_rsrps) == 0 and len(nr_rsrps) == 0:
            continue
        if max(lte_rsrps) >= max(nr_rsrps) if len(nr_rsrps) > 0 else True:
            current_sinr = lte_sinrs[np.argmax(lte_rsrps)] if len(lte_rsrps) > 0 else -float('inf')
        else:
            current_sinr = nr_sinrs[np.argmax(nr_rsrps)] if len(nr_rsrps) > 0 else -float('inf')
        initial_features.append([uav_path_x[0], uav_path_y[0], current_sinr])
    if initial_features:
        q_agent.fit_scaler(np.array(initial_features))
    q_agent.epsilon = epsilon
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
    per_step_times = [] # Collect per-step times for inference complexity
    uav_total_times = [] # Time per UAV
    num_total_steps = num_uavs * num_steps # Approximate total steps
    start_time = time.time()
    for uav_idx in range(num_uavs):
        uav_start = time.time() # Start per-UAV timer
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
        # Initial cell selection (time as first step)
        step_start = time.time()
        lte_rsrps, lte_sinrs, lte_noise_dbm, lte_interference = get_sinr(uav_path_x[0], uav_path_y[0], coords_4g, lte_config['height'], lte_config['ptx'], lte_config['freq'], lte_gain, lte_config['noise'], uav_z, lte_freqs)
        nr_rsrps, nr_sinrs, nr_noise_dbm, nr_interference = get_sinr(uav_path_x[0], uav_path_y[0], coords_5g, nr_config['height'], nr_config['ptx'], nr_config['freq'], nr_gain, nr_config['noise'], uav_z, nr_freqs)
        step_end = time.time()
        per_step_times.append(step_end - step_start) # Include initial step
       
        if len(lte_rsrps) == 0 and len(nr_rsrps) == 0:
            print(f"UAV {uav_idx + 1}: No base stations available at initial position.")
            continue
        if max(lte_rsrps) >= max(nr_rsrps) if len(nr_rsrps) > 0 else True:
            current_system, current_rsrp, current_bs_idx = 'LTE', max(lte_rsrps) if len(lte_rsrps) > 0 else -float('inf'), np.argmax(lte_rsrps) if len(lte_rsrps) > 0 else 0
            current_sinr = lte_sinrs[current_bs_idx]
        else:
            current_system, current_rsrp, current_bs_idx = 'NR', max(nr_rsrps), np.argmax(nr_rsrps)
            current_sinr = nr_sinrs[current_bs_idx]
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
            step_start = time.time()
            x, y = uav_path_x[t], uav_path_y[t]
            lte_rsrps, lte_sinrs, lte_noise_dbm, lte_interference = get_sinr(x, y, coords_4g, lte_config['height'], lte_config['ptx'], lte_config['freq'], lte_gain, lte_config['noise'], uav_z, lte_freqs)
            nr_rsrps, nr_sinrs, nr_noise_dbm, nr_interference = get_sinr(x, y, coords_5g, nr_config['height'], nr_config['ptx'], nr_config['freq'], nr_gain, nr_config['noise'], uav_z, nr_freqs)
            # Construct feature vector X_t
            lte_rsrp_max = max(lte_rsrps) if lte_rsrps.size > 0 else -float('inf')
            nr_rsrp_max = max(nr_rsrps) if nr_rsrps.size > 0 else -float('inf')
            noise_dbm = lte_noise_dbm if current_system == 'LTE' else nr_noise_dbm
            interference_dbm = lte_interference[current_bs_idx] if current_system == 'LTE' else nr_interference[current_bs_idx]
            interference_dbm = interference_dbm if interference_dbm != -float('inf') else 0
            eta = eta_from_sinr(current_sinr)
            bw_hz = lte_config['bw_hz'] if current_system == 'LTE' else nr_config['bw_hz']
            R = bw_hz * eta / 1e6
            L_net = (1000 + 4000) / (R * 1e6) if R > 0 else float('inf')
            feature_vector = np.array([
                x, y, current_bs_idx, current_rsrp, current_sinr,
                lte_rsrp_max, nr_rsrp_max, noise_dbm, interference_dbm,
                R, L_net, 1 if current_system == 'LTE' else 0, 1 if current_system == 'NR' else 0
            ])
            current_state = q_agent.get_state(x, y, current_system, current_bs_idx, current_sinr, current_rsrp)
            action = q_agent.choose_action(current_state, lte_rsrps, nr_rsrps, current_system, current_bs_idx, feature_vector)
            next_system, next_bs_idx = action
            is_handover = (action[0] != 'stay') and ((next_system != current_system) or (next_bs_idx != current_bs_idx))
            sinr_db = lte_sinrs[next_bs_idx] if next_system == 'LTE' else nr_sinrs[next_bs_idx]
            reward = q_agent.get_reward(sinr_db, is_handover)
            if is_handover:
                current_system, current_bs_idx = next_system, next_bs_idx
                current_rsrp = lte_rsrps[current_bs_idx] if current_system == 'LTE' else nr_rsrps[current_bs_idx]
                current_sinr = lte_sinrs[current_bs_idx] if current_system == 'LTE' else nr_sinrs[current_bs_idx]
                handover_times.append(t * measurement_interval)
                if current_system == 'LTE':
                    lte_handover_count += 1
                else:
                    nr_handover_count += 1
            next_state = q_agent.get_state(x, y, current_system, current_bs_idx, current_sinr, current_rsrp)
            actions = q_agent.get_actions(lte_rsrps, nr_rsrps, current_system, current_bs_idx)
            q_agent.update_q_table(current_state, action, reward, next_state, actions)
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
            step_end = time.time()
            per_step_times.append(step_end - step_start)
        uav_end = time.time()
        uav_total_times.append(uav_end - uav_start) # Per-UAV time
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
    # Save Q-table
    if save_q_table:
        with open(f'q_table{suffix}.pkl', 'wb') as f:
            pickle.dump(dict(q_agent.q_table), f)
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
            L_net = (E_S_bits + E_Q_bits) / (R * 1e6) if R > 0 else float('inf')
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
    # Generate CSV Datasets
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
    # Print Results
    simulation_end = time.time()
    total_simulation_time = simulation_end - simulation_start
    avg_per_step_time = np.mean(per_step_times) if per_step_times else 0
    total_per_step_time = np.sum(per_step_times)
    avg_uav_time = np.mean(uav_total_times) if uav_total_times else 0
    print(f"\n=== Training Duration and Inference Complexity Report for {total_time_s}s Run ===")
    print(f"Total Simulation Runtime: {total_simulation_time:.2f} seconds")
    print(f"Time per UAV: {np.mean(uav_total_times):.2f} seconds (std: {np.std(uav_total_times):.2f}s)")
    print(f"Average Inference Time per Step: {avg_per_step_time * 1000:.2f} ms")
    print(f"Total Time in Steps: {total_per_step_time:.2f} seconds ({total_per_step_time / total_simulation_time * 100:.1f}% of total)")
    print(f"Total Operations Estimate: ~{num_total_steps * 3850} (based on O(num_bs * num_beams))")
    print("Note: Per-step includes 2x get_sinr calls + XGB predict + Q-Learning choose/update logic. Run on your hardware for exact values.")
    print(f"\nSimulation time for {total_time_s} seconds: {total_simulation_time:.2f} seconds")
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
    