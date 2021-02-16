import pandas as pd


def height2lat(height):
    return height / 110.574


def width2lng(width):
    return width / 111.320 / 0.99974


def in_boundary(lat, lng, b):
    return b['min_lng'] < lng < b['max_lng'] and b['min_lat'] < lat < b['max_lat']


def main():
    lat_size, lng_size = height2lat(grid_height), width2lng(grid_width)

    lat_grid_num = int((boundary['max_lat'] - boundary['min_lat']) / lat_size) + 1
    lng_grid_num = int((boundary['max_lng'] - boundary['min_lng']) / lng_size) + 1

    trajectories = pd.read_csv("{}/{}.csv".format(data_dir, data_name), header=0, index_col="TRIP_ID")
    processed_trajectories = []

    shortest, longest = 20, 1200
    total_traj_num = len(trajectories)
    for i, (idx, traj) in enumerate(trajectories.iterrows()):
        if i % 10000 == 0:
            print("Complete: {}; Total: {}".format(i, total_traj_num))
        grid_seq = []
        valid = True
        polyline = eval(traj["POLYLINE"])
        if shortest <= len(polyline) <= longest:
            for lng, lat in polyline:
                if in_boundary(lat, lng, boundary):
                    grid_i = int((lat - boundary['min_lat']) / lat_size)
                    grid_j = int((lng - boundary['min_lng']) / lng_size)
                    grid_seq.append((grid_i, grid_j))
                else:
                    valid = False
                    break
            if valid:
                processed_trajectories.append(grid_seq)
    print("Valid trajectory num:", len(processed_trajectories))
    print("Grid size:", (lat_grid_num, lng_grid_num))

    fout = open("{}/processed_{}.csv".format(data_dir, data_name), 'w')
    for traj in processed_trajectories:
        fout.write("[")
        for i, j in traj[:-1]:
            fout.write("%s, " % str(i * lng_grid_num + j))
        fout.write("%s]\n" % str(traj[-1][0] * lng_grid_num + traj[-1][1]))


if __name__ == '__main__':
    data_dir = '../data'
    data_name = "porto"
    grid_height, grid_width = 0.1, 0.1
    boundary = {'min_lat': 41.140092, 'max_lat': 41.185969, 'min_lng': -8.690261, 'max_lng': -8.549155}
    main()
