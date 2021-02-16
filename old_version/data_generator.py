import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict


def pad_and_mask(batch_x):
    max_len = max(len(x) for x in batch_x)
    batch_mask = [[1] * len(x) + [0] * (max_len - len(x)) for x in batch_x]
    batch_x = [x + [0] * (max_len - len(x)) for x in batch_x]
    return batch_x, batch_mask


class DataGenerator:
    def __init__(self, args):
        print("Loading data...")

        self.trajectories = sorted([
            eval(eachline) for eachline in open(args.data_filename.format("_train"), 'r').readlines()
        ], key=lambda k: len(k))
        self.total_traj_num = len(self.trajectories)

        self.val_trajectories = sorted([
            eval(eachline) for eachline in open(args.data_filename.format("_val"), 'r').readlines()
        ], key=lambda k: len(k))
        self.val_traj_num = len(self.val_trajectories)

        self.real_trajectories = self.trajectories[:]
        self.real_val_trajectories = self.val_trajectories[:]

        self.args = args
        print("{} trajectories loading complete.".format(self.total_traj_num))

        self.traj_sd = {idx: [traj[0], traj[-1]] for idx, traj in enumerate(self.trajectories)}
        self.val_traj_sd = {idx: [traj[0], traj[-1]] for idx, traj in enumerate(self.val_trajectories)}

        self.map_size = args.map_size
        self.sd_index = self.construct_sd_index()
        self.sd_ids = {sd: i for i, sd in enumerate(self.sd_index.keys())}
        print("Totally {} sd-pairs".format(len(self.sd_ids)))

        self.traj_sd_cluster = self.traj_sd
        self.val_traj_sd_cluster = self.val_traj_sd

    def make_model_dir(self):
        if hasattr(self.args, 'rnn_size'):
            model_dir = './models/{}_{}_{}/'.format(
                self.args.model_type, self.args.x_latent_size, self.args.rnn_size)
        elif hasattr(self.args, 'x_latent_size'):
            model_dir = './models/{}_{}/'.format(
                self.args.model_type, self.args.x_latent_size)
        else:
            model_dir = './models/{}/'.format(self.args.model_type)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        return model_dir

    def construct_sd_index(self):
        sd_index = defaultdict(list)
        for idx, traj in enumerate(self.trajectories):
            sd_index[(traj[0], traj[-1])].append(idx)
        return sd_index

    def inject_outliers(self, otype, ratio=0.05, level=2, point_prob=0.3, vary=False):
        # inject in training data
        selected_idx = np.random.randint(0, self.total_traj_num, size=int(self.total_traj_num * ratio))
        # selected_idx = np.concatenate([np.random.choice(tids, size=int(len(tids) * ratio), replace=False)
        #                                for tids in self.sd_index.values()], axis=0)
        if otype == 'random':
            outliers = self.perturb_batch([self.trajectories[idx] for idx in selected_idx],
                                          level=level, prob=point_prob)
        elif otype == 'pan':
            outliers = self.pan_batch([self.trajectories[idx] for idx in selected_idx],
                                      level=level, prob=point_prob, vary=vary)
        else:
            outliers = None

        for i, idx in enumerate(selected_idx):
            self.trajectories[idx] = outliers[i]

        self.outliers = dict(zip(selected_idx, outliers))
        model_dir = self.make_model_dir()
        with open(model_dir + 'porto_outliers_{}.pkl'.format(self.args.model_type), 'wb') as fp:
            pickle.dump(self.outliers, fp)
        print("{} outliers injection complete.".format(len(outliers)))

        # inject in validation data
        selected_idx = np.random.randint(0, len(self.val_trajectories),
                                         size=int(len(self.val_trajectories) * ratio))
        if otype == 'random':
            outliers = self.perturb_batch([self.val_trajectories[idx] for idx in selected_idx],
                                          level=level, prob=point_prob)
        elif otype == 'pan':
            outliers = self.pan_batch([self.val_trajectories[idx] for idx in selected_idx],
                                          level=level, prob=point_prob)
        else:
            outliers = None
        for i, idx in enumerate(selected_idx):
            self.val_trajectories[idx] = outliers[i]

    def load_outliers(self):
        model_dir = self.make_model_dir()
        with open(model_dir + 'porto_outliers_{}.pkl'.format(self.args.model_type), 'rb') as fp:
            if sys.version[0] == '2':
                self.outliers = outliers = pickle.load(fp)
            else:
                self.outliers = outliers = pickle.load(fp, encoding='latin1')
            for idx, o in outliers.items():
                self.trajectories[idx] = o
        print("{} outliers loading complete.".format(len(outliers)))

    def load_external_outliers(self, path):
        with open(path, 'rb') as fp:
            if sys.version[0] == '2':
                self.outliers = outliers = pickle.load(fp)
            else:
                self.outliers = outliers = pickle.load(fp, encoding='latin1')
            for idx, o in outliers.items():
                self.trajectories[idx] = o
        print("{} outliers loading complete.".format(len(outliers)))

    def next_batch(self, batch_size, partial_ratio=1.0, sd=False):
        anchor_idx = np.random.randint(0, self.total_traj_num)
        shortest_idx = max(0, anchor_idx - batch_size * 2)
        longest_idx = min(self.total_traj_num, anchor_idx + batch_size * 2)
        batch_idx = np.random.randint(shortest_idx, longest_idx, size=batch_size)
        batch_trajectories = []
        batch_s, batch_d = [], []
        for tid in batch_idx:
            partial = int(len(self.trajectories[tid]) * partial_ratio)
            batch_trajectories.append(self.trajectories[tid][:partial])
            batch_s.append(self.traj_sd_cluster[tid][0])
            batch_d.append(self.traj_sd_cluster[tid][1])
        batch_seq_length = [len(traj) for traj in batch_trajectories]
        batch_x, batch_mask = pad_and_mask(batch_trajectories)
        if "sd" in self.args.model_type or sd is True:
            return [batch_x, batch_mask, batch_seq_length], [batch_s, batch_d]
        else:
            return [batch_x, batch_mask, batch_seq_length]

    def iterate_all_data(self, batch_size, partial_ratio=1.0, purpose='train'):
        if purpose == 'train':
            trajectories = self.trajectories
            traj_num = self.total_traj_num
            traj_sd_cluster = self.traj_sd_cluster
        elif purpose == "val":
            trajectories = self.val_trajectories
            traj_num = self.val_traj_num
            traj_sd_cluster = self.val_traj_sd_cluster
        else:
            trajectories = None
            traj_num = None
            traj_sd_cluster = None

        for batch_idx in range(0, traj_num, batch_size):
            batch_trajectories = []
            batch_s, batch_d = [], []
            for tid in range(batch_idx, min(batch_idx + batch_size, traj_num)):
                partial = int(len(trajectories[tid]) * partial_ratio)
                batch_trajectories.append(trajectories[tid][:partial])
                batch_s.append(traj_sd_cluster[tid][0])
                batch_d.append(traj_sd_cluster[tid][1])
            batch_seq_length = [len(traj) for traj in batch_trajectories]
            batch_x, batch_mask = pad_and_mask(batch_trajectories)
            if "sd" in self.args.model_type:
                yield [batch_x, batch_mask, batch_seq_length], [batch_s, batch_d]
            else:
                yield [batch_x, batch_mask, batch_seq_length]

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

    def pan_batch(self, batch_x, level, prob, vary=False):
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

    def visualize(self, traj, c, alpha=1.0, lw=3, ls='-'):
        map_size = self.map_size
        gx, gy = [], []
        for p in traj:
            if not p == 0:
                gy.append(int(p // map_size[1]))
                gx.append(int(p % map_size[1]))
        plt.plot(gx, gy, color=c, linestyle=ls, lw=lw, alpha=alpha)

    def visualize_outlier_example(self, num, defult_idx=None):
        outlier_items = list(self.outliers.items())
        np.random.shuffle(outlier_items)
        if defult_idx is not None:
            selected_outliers = [(idx, self.outliers[idx]) for idx in defult_idx]
        else:
            selected_outliers = outlier_items[:num]
        for idx, o in selected_outliers:
            print(idx)
            self.visualize(o, c='r', alpha=1.0)
            self.visualize(self.real_trajectories[idx], c='b', alpha=0.5)
        # plt.xlim(0, self.args.map_size[1])
        # plt.ylim(0, self.args.map_size[0])
        plt.show()

    def __spatial_augmentation(self):
        self.trajectories = [self.perturb_batch([traj], level=1, prob=0.3)[0]
                             for traj in self.real_trajectories]
        self.val_trajectories = [self.perturb_batch([traj], level=1, prob=0.3)[0]
                                 for traj in self.real_val_trajectories]

    def __sd_clustering(self):
        from sklearn.cluster import KMeans
        args = self.args

        traj_sd = np.array(list(self.traj_sd.values())).reshape(-1)
        sd_locs = np.array([[int(p // args.map_size[1]), int(p % args.map_size[1])] for p in traj_sd])
        val_traj_sd = np.array(list(self.val_traj_sd.values())).reshape(-1)
        val_sd_locs = np.array([[int(p // args.map_size[1]), int(p % args.map_size[1])] for p in val_traj_sd])

        self.kmeans = kmeans = KMeans(n_clusters=args.sd_cluster_num).fit(sd_locs)
        sd_clusters = kmeans.transform(sd_locs).reshape([-1, 2, args.sd_cluster_num]).tolist()
        self.traj_sd_cluster = dict(zip(self.traj_sd.keys(), sd_clusters))
        val_sd_clusters = kmeans.transform(val_sd_locs).reshape([-1, 2, args.sd_cluster_num]).tolist()
        self.val_traj_sd_cluster = dict(zip(self.val_traj_sd.keys(), val_sd_clusters))
        self.__save_clustering()

    def __save_clustering(self):
        model_dir = self.make_model_dir()
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        with open(model_dir + 'sd_clusters.pkl', 'wb') as fp:
            pickle.dump([self.kmeans,
                         self.traj_sd_cluster,
                         self.val_traj_sd_cluster], fp)

    def __load_clustering(self):
        model_dir = self.make_model_dir()
        with open(model_dir + 'sd_clusters.pkl', 'rb') as fp:
            self.kmeans, self.traj_sd_cluster, self.val_traj_sd_cluster = pickle.load(fp, encoding='latin1')