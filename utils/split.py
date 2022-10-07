import random

def split(path, train_size=0.8):
    dct = {}
    with open(path) as rPtr:
        for line in rPtr:
            infos = line.rstrip().split(' ')
            if infos[1] not in dct.keys():
                dct[infos[1]] = []
            dct[infos[1]].append(
                infos[0].split('/')[-1].split('.')[0])

    trains, vals = [], []
    for key in dct.keys():
        random.shuffle(dct[key])

        split_idx = int(len(dct[key]) * train_size)

        train = dct[key][:split_idx]
        val = dct[key][split_idx:]

        trains.extend(train)
        vals.extend(val)

    with open('./data/train.txt', 'w') as wPtr:
        for trian_item in trains:
            wPtr.write(trian_item + '\n')

    with open('./data/val.txt', 'w') as wPtr:
        for val_item in vals:
            wPtr.write(val_item + '\n')
