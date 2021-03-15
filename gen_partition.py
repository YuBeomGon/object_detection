import numpy as np


def generate_partition(out_path, labels_info_path, train_ratio=0.7, oversampling=False):
    labels_info = np.load(labels_info_path, allow_pickle=True, encoding='latin1').item()
    
    # Delete ID when labels not exist.
    valid_image_IDs = []
    for k, v in labels_info.items():
        if len(v) != 0:
            valid_image_IDs.append(k)
    
    # Generate Partition (Train / Test)
    num_of_data = len(valid_image_IDs)
    num_train = np.round(num_of_data * train_ratio).astype(int)
    num_test = num_of_data - num_train
    ratio_train = np.round(np.float(num_train) / np.float(num_of_data) * 100).astype(int)
    ratio_test = np.round(np.float(num_test) / np.float(num_of_data) * 100).astype(int)

    print("{} for train and {} for test".format(num_train, num_test))
    print("{}% for train and {}% for test in full dataset".format(ratio_train, ratio_test))

    ## Random choice for splitting
    ### Shuffle image IDs
    np.random.shuffle(valid_image_IDs)

    idx_train = np.random.choice(num_of_data, size=num_train, replace=False)
    idx_test = np.setdiff1d(list(range(num_of_data)), idx_train)
    idx_train.sort()
    idx_test.sort()


    partition = {}
    train_IDs = []
    test_IDs = []
    for train_i in idx_train:
        train_ID = valid_image_IDs[train_i]
        train_IDs.append(train_ID)

    for test_i in idx_test:
        test_ID = valid_image_IDs[test_i]
        test_IDs.append(test_ID)

    partition['train'] = train_IDs
    partition['test'] = test_IDs

    np.save(out_path, partition)


if __name__ == '__main__':
    out_path = './data/partition.npy'
    labels_info_path = './data/labels_info.npy'
    
    generate_partition(out_path=out_path, 
                       labels_info_path=labels_info_path,
                       train_ratio=0.7, 
                       oversampling=False)