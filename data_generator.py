import os
import sys
import pickle
import numpy as np
# import matplotlib.pyplot as plt

from collections import defaultdict
import tensorflow as tf



class DataGenerator:
    def __init__(self, args):
        print("Loading data...")
        self.args = args
        self.map_size = args.map_size

        self.train_trajectories, self.train_sd, self.train_traj_num = self.build_dataset('train')
        self.val_trajectories, self.val_sd, self.val_traj_num = self.build_dataset('val')

        self.data_iterator = {'train': lambda: self.iterate_data(data_type='train'),
                              'val': lambda: self.iterate_data(data_type='val')}

    def build_dataset(self, data_type):
        args = self.args
        data_name = args.data_filename.split('.')
        data_name[-2] += "_{}".format(data_type)
        data_name = ".".join(data_name)
        trajectories = sorted([
            eval(eachline) for eachline in open(data_name, 'r').readlines()
        ], key=lambda k: len(k))
        traj_num = len(trajectories)
        print("{} {} trajectories loading complete.".format(traj_num, data_type))
        # traj_sd = {idx: [traj[0], traj[-1]] for idx, traj in enumerate(trajectories)}

        traj_sd = defaultdict(list)
        for idx, traj in enumerate(trajectories):
            traj_sd[(traj[0], traj[-1])].append(idx)

        return trajectories, traj_sd, traj_num

    def inject_outliers(self, otype, data_type, ratio=0.05, level=2, point_prob=0.3, vary=False):
        # inject in training data
        if data_type == "train":
            traj_num = self.train_traj_num
            trajectories = self.train_trajectories
        elif data_type == "val":
            traj_num = self.val_traj_num
            trajectories = self.val_trajectories
        else:
            raise ValueError("data_type is not 'train' or 'val'.")

        self.outlier_idx = selected_idx = np.random.randint(0, traj_num, size=int(traj_num * ratio))
        if otype == 'random':
            outliers = self.perturb_batch([trajectories[idx] for idx in selected_idx],
                                          level=level, prob=point_prob)
        elif otype == 'shift':
            outliers = self.shift_batch([trajectories[idx] for idx in selected_idx],
                                      level=level, prob=point_prob, vary=vary)
        else:
            raise ValueError("otype is not 'random' or 'shift'.")

        for i, idx in enumerate(selected_idx):
            trajectories[idx] = outliers[i]

        out_filename = 'porto_outliers.pkl'
        with open('./data/' + out_filename, 'wb') as fp:
            pickle.dump(dict(zip(selected_idx, outliers)), fp)

        print("{} outliers injection into {} is completed.".format(len(outliers), data_type))

    def load_outliers(self, filename, data_type):
        if data_type == "train":
            trajectories = self.train_trajectories
        elif data_type == "val":
            trajectories = self.val_trajectories

        with open('./data/' + filename, 'rb') as fp:
            idx_outliers = pickle.load(fp)
        for idx, o in idx_outliers.items():
            trajectories[idx] = o

        self.outlier_idx = list(idx_outliers.keys())
        
    def pad_and_mask(self, batch_x):
        max_len = max(len(x) for x in batch_x)
        batch_mask = [[1] * len(x) + [0] * (max_len - len(x)) for x in batch_x]
        batch_x = [x + [0] * (max_len - len(x)) for x in batch_x]
        return batch_x, batch_mask

    def next_batch(self, batch_size, partial_ratio=1.0, sd=False):
        anchor_idx = np.random.randint(0, self.train_traj_num)
        shortest_idx = max(0, anchor_idx - batch_size * 2)
        longest_idx = min(self.train_traj_num, anchor_idx + batch_size * 2)
        batch_idx = np.random.randint(shortest_idx, longest_idx, size=batch_size)
        batch_trajectories = []
        batch_s, batch_d = [], []
        for tid in batch_idx:
            partial = int(len(self.train_trajectories[tid]) * partial_ratio)
            batch_trajectories.append(self.train_trajectories[tid][:partial])
            # batch_s.append(self.traj_sd_cluster[tid][0])
            # batch_d.append(self.traj_sd_cluster[tid][1])
        batch_seq_length = [len(traj) for traj in batch_trajectories]
        batch_x, batch_mask = self.pad_and_mask(batch_trajectories)
        # if "sd" in self.args.model or sd is True:
        #     return [batch_x, batch_mask, batch_seq_length], [batch_s, batch_d]
        # else:
        return map(tf.convert_to_tensor, [batch_x, batch_mask, batch_seq_length])

    def iterate_data(self, data_type='val', partial_ratio=1.0, sd=False):
        batch_size = self.args.batch_size

        if data_type == 'train':
            traj_num = self.train_traj_num
            trajectories = self.train_trajectories + self.train_trajectories[:batch_size]
        elif data_type == 'val':
            traj_num = self.val_traj_num
            trajectories = self.val_trajectories + self.val_trajectories[:batch_size]

        for shortest_idx in range(0, traj_num, batch_size):
            longest_idx = shortest_idx + batch_size
            batch_trajectories = []
            for tid in range(shortest_idx, longest_idx):
                partial = int(len(trajectories[tid]) * partial_ratio)
                batch_trajectories.append(trajectories[tid][:partial])
                # batch_s.append(self.traj_sd_cluster[tid][0])
                # batch_d.append(self.traj_sd_cluster[tid][1])
            batch_seq_length = [len(traj) for traj in batch_trajectories]
            batch_x, batch_mask = self.pad_and_mask(batch_trajectories)
            # if "sd" in self.args.model or sd is True:
            #     return [batch_x, batch_mask, batch_seq_length], [batch_s, batch_d]
            # else:
            yield batch_x, batch_mask, batch_seq_length

    def _perturb_point(self, point, level, offset=None):
        map_size = self.map_size
        x, y = int(point // map_size[1]), int(point % map_size[1])
        if offset is None:
            offset = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
            x_offset, y_offset = offset[np.random.randint(0, len(offset))]
        else:
            x_offset, y_offset = offset
        if 0 <= x + x_offset * level < map_size[0] and 0 <= y + y_offset * level < map_size[1]:
            x += x_offset * level
            y += y_offset * level
        return int(x * map_size[1] + y)

    def perturb_batch(self, batch_x, level, prob):
        noisy_batch_x = []
        for traj in batch_x:
            noisy_batch_x.append([traj[0]] + [self._perturb_point(p, level)
                                 if not p == 0 and np.random.random() < prob else p
                                 for p in traj[1:-1]] + [traj[-1]])
        return noisy_batch_x

    def shift_batch(self, batch_x, level, prob, vary=False):
        map_size = self.map_size
        noisy_batch_x = []
        if vary:
            level += np.random.randint(-2, 3)
            if np.random.random() > 0.5:
                prob += 0.2 * np.random.random()
            else:
                prob -= 0.2 * np.random.random()
        for traj in batch_x:
            anomaly_len = int((len(traj) - 2) * prob)
            anomaly_st_loc = np.random.randint(1, len(traj) - anomaly_len - 1)
            anomaly_ed_loc = anomaly_st_loc + anomaly_len

            offset = [int(traj[anomaly_st_loc] // map_size[1]) - int(traj[anomaly_ed_loc] // map_size[1]),
                      int(traj[anomaly_st_loc] % map_size[1]) - int(traj[anomaly_ed_loc] % map_size[1])]
            if offset[0] == 0: div0 = 1
            else: div0 = abs(offset[0])
            if offset[1] == 0: div1 = 1
            else: div1 = abs(offset[1])

            if np.random.random() < 0.5:
                offset = [-offset[0] / div0, offset[1] / div1]
            else:
                offset = [offset[0] / div0, -offset[1] / div1]

            noisy_batch_x.append(traj[:anomaly_st_loc] +
                                 [self._perturb_point(p, level, offset) for p in traj[anomaly_st_loc:anomaly_ed_loc]] +
                                 traj[anomaly_ed_loc:])
        return noisy_batch_x
