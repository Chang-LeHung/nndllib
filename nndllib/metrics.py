
import numpy as _np
from tensorflow.keras.applications.inception_v3 import InceptionV3 as _InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as _preprocess_input
import tensorflow as _tf
import torch as _torch
import sys as _sys
from torchvision import transforms as _transforms
from torchvision import datasets as _datasets
from torch.utils.data import DataLoader as _DataLoader
from .utils import extract_from_dataset as _extract_from_dataset



# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(path=None, images=None, dataset=None,  n_split=10, eps=1E-16, batch_size=2048,
                              device="GPU:0", topK=None):
    if images is None:
        if dataset is None:
            dataset = _datasets.ImageFolder(root=path,
                                       transform=_transforms.Compose([
                                           _transforms.Resize((299, 299)),
                                           _transforms.ToTensor(),
                                       ]))
        dataset = _DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=8)
        images = _extract_from_dataset(dataset, 0, topK=topK)
        # load inception v3 model
        with _tf.device("/CPU"):
            images = images.permute(0, 2, 3, 1).mul(255).data.numpy().astype(_np.uint8)
    model = _InceptionV3()
    # enumerate splits of images/predictions
    scores = list()
    n_part = images.shape[0] // n_split
    print("Starting to calculate inception score...", file=_sys.stderr)
    for i in range(n_split):
        # retrieve images
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        subset = images[ix_start:ix_end]
        # convert from uint8 to float32
        subset = subset.astype('float32')
        # scale images to the required size
        # pre-process images, scale to [-1,1]
        with _tf.device(device):
            subset = _preprocess_input(subset)
            # predict p(y|x)
            p_yx = model.predict(subset)
        print(f"step[{i + 1}/{n_split}] 第 {1 + i} 轮计算完成", file=_sys.stderr)
        # calculate p(y)
        p_y = _np.expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (_np.log(p_yx + eps) - _np.log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = _np.mean(sum_kl_d)
        # undo the log
        is_score = _np.exp(avg_kl_d)
        # store
        scores.append(is_score)
    # average across images
    is_avg, is_std = _np.mean(scores), _np.std(scores)
    return is_avg, is_std


if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser()

    parse.add_argument("--path", type=str)
    args = parse.parse_args()
    print(calculate_inception_score(args.path))

