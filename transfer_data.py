import numpy as np
import os


def transfer_data(in_dir, out_dir):
    data_dir = os.path.join(in_dir, 'train_data.npy')
    label_dir = os.path.join(in_dir, 'train_label.npy')

    data = np.load(data_dir)  # 2294 3 2500 25 1
    label = np.load(label_dir, allow_pickle=True)  # 2294
    ans = {'split': {}, 'annotations': []}
    N, C, T, V, M = data.shape
    sep = int(N * 0.9)
    train = []
    val = []
    for i in range(0, N):
        if i < sep:
            train.append(str(i))
        else:
            val.append(str(i))
    split = dict(
        train=train,
        val=val
    )
    for i in range(data.shape[0]):
        frame = data[i]
        frame = np.transpose(frame, (3, 1, 2, 0))
        keypoint = frame[..., :2]
        keypoint_score = frame[..., 2]
        ans['annotations'].append({'frame_dir': str(i),
                                   'label': label[i],
                                   'img_shape': (1080, 720),
                                   'original_shape': (1080, 720),
                                   'total_frames': 2500,
                                   'keypoint': keypoint,
                                   'keypoint_score': keypoint_score})
    ans['split'] = split
    import pickle
    with open(os.path.join(out_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(ans, f)


if __name__ == '__main__':
    transfer_data('data', 'data')
