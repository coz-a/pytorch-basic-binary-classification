from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import cv2

NORMALIZE_PARAMS = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class CustomDataSet(Dataset):

    def __init__(self, image_dir, annot_file, transforms=None):
        self.image_files = sorted(Path(image_dir).glob("*.jpg"))
        self.annots = pd.read_csv(annot_file, index_col=0).iloc[:, 0]
        self.transforms = transforms

    def __getitem__(self, i):
        image_file = self.image_files[i]
        image = cv2.imread(str(image_file))[..., ::-1]  # bgr to rgb
        image_id = image_file.stem
        label = self.annots[image_id]
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

    def get_all_labels(self):
        return [self.annots[_.stem] for _ in self.image_files]


class Transformed(Dataset):

    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        return self.transforms(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)
