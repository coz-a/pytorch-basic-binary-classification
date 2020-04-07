class Partial:

    def __init__(self, transform, src_key="image", dst_key=None):
        self.transform = transform
        self.src_key = src_key
        self.dst_key = src_key if dst_key is None else dst_key

    def __call__(self, sample):
        sample_ = sample.copy()
        sample_[self.dst_key] = self.transform(sample[self.src_key])
        return sample_
