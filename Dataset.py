import torch.utils.data as dt
import random
import torch
import numpy as np


class PicDataSet(dt.IterableDataset):
    def __init__(self, pic_data, y_pic_data, num_each_pic, size):
        super(PicDataSet).__init__()
        self.pic_data = pic_data
        self.y_pic_data = y_pic_data
        self.num_each_pic = num_each_pic
        self.size = size
        self.current_position = 0
        self.pic_already_generated = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.record_next()
        if self.current_position >= len(self.pic_data):
            raise StopIteration
        take_current_pic = self.pic_data[self.current_position]
        take_current_y_pic = self.y_pic_data[self.current_position]
        start_pos = random.randint(0, len(take_current_pic) - self.size)
        return take_current_pic[start_pos:start_pos + self.size], take_current_y_pic[start_pos:start_pos + self.size]

    def __len__(self):
        return len(self.pic_data) * self.num_each_pic

    def record_next(self):
        if self.pic_already_generated >= self.num_each_pic:
            self.pic_already_generated = 1
            self.current_position += 1
        else:
            self.pic_already_generated += 1


if __name__ == "__main__":
    dataset = np.array([
        [np.random.rand(1, 128, 128) for i in range(5)],
        [np.random.rand(1, 128, 128) for i in range(5)]
    ])
    dataset = torch.tensor(dataset)
    ds = PicDataSet(dataset, dataset, 3, 3)
    dl = dt.DataLoader(ds, batch_size=2)
    for batch, (X, y) in enumerate(dl):
        print(batch)
        print(X.shape)
        print(y.shape)
