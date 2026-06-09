import numpy as np
import torch


class Cutout:
    """Mask one or more square regions in a tensor image."""

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        height = img.size(1)
        width = img.size(2)
        mask = np.ones((height, width), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(height)
            x = np.random.randint(width)

            y1 = np.clip(y - self.length // 2, 0, height)
            y2 = np.clip(y + self.length // 2, 0, height)
            x1 = np.clip(x - self.length // 2, 0, width)
            x2 = np.clip(x + self.length // 2, 0, width)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask


class RABatchSampler:
    """Repeated-augmentation sampler.

    Each yielded batch repeats the same unique image index several times so the
    dataset transform can produce multiple augmented views.
    """

    def __init__(self, n, batch_size, repeats=3):
        self.n = n
        self.batch_size = batch_size
        self.repeats = repeats
        self.num_unique = batch_size // repeats

    def __len__(self):
        return self.n // self.num_unique

    def __iter__(self):
        indices = torch.randperm(self.n).tolist()
        for i in range(0, len(indices), self.num_unique):
            chunk = indices[i : i + self.num_unique]
            if len(chunk) < self.num_unique:
                continue
            batch = []
            for idx in chunk:
                batch.extend([idx] * self.repeats)
            yield batch
