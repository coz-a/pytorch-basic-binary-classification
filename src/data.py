from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import cv2

NORMALIZE_PARAMS = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class ImagewoofBinaryDataSet(Dataset):

    def __init__(self, image_dir, neg_classes=["n02086240"], pos_classes=["n02087394"], transforms=None):
        neg_files = sum([
            sorted((Path(image_dir) / _).glob("*.JPEG"))
            for _ in neg_classes
        ], [])
        pos_files = sum([
            sorted((Path(image_dir) / _).glob("*.JPEG"))
            for _ in pos_classes
        ], [])
        self.image_files = neg_files + pos_files
        self.labels = np.array([0] * len(neg_files) + [1] * len(pos_files))
        self.transforms = transforms

    def __getitem__(self, i):
        image_file = self.image_files[i]
        image = cv2.imread(str(image_file))[..., ::-1]  # bgr to rgb
        image_id = image_file.stem
        label = self.labels[i]
        sample = dict(
            image_id=image_id,
            image=image,
            label=label)
        if self.transforms is None:
            return sample
        else:
            return self.transforms(sample)

    def __len__(self):
        return len(self.image_files)


def oversample(labels):
    classes = np.unique(labels)
    indices = [np.where(labels == _)[0] for _ in classes]
    n_largest = max(len(_) for _ in indices)
    return np.concatenate(
        [_oversample(_, n_largest) for _ in indices]
    )


def _oversample(samples, n_target):
    n_samples = len(samples)
    if n_samples >= n_target:
        return samples
    quot = n_target // n_samples
    rem = n_target % n_samples
    return np.concatenate([samples] * quot + [samples[:rem]])


class Transformed(Dataset):

    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        return self.transforms(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)
