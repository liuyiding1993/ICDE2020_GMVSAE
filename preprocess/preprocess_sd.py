from collections import defaultdict


def main():
    f = open("{}/processed_{}.csv".format(data_dir, data_name), 'r').readlines()
    sd_cnt = defaultdict(list)
    for eachline in f:
        traj = eval(eachline)
        s, d = traj[0], traj[-1]
        sd_cnt[(s, d)].append(eachline)

    fout_train = open("{}/processed_{}_train.csv".format(data_dir, data_name), 'w')
    fout_test = open("{}/processed_{}_val.csv".format(data_dir, data_name), 'w')
    for trajs in sd_cnt.values():
        if len(trajs) >= min_sd_traj_num:
            train_trajs, test_trajs = trajs[:-test_traj_num], trajs[-test_traj_num:]
            for traj in train_trajs:
                fout_train.write(traj)
            for traj in test_trajs:
                fout_test.write(traj)
    

if __name__ == '__main__':
    data_dir = '../data'
    data_name = "porto"
    min_sd_traj_num = 25
    test_traj_num = 5
    main()
