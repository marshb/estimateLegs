import os
import numpy as np
from scipy import misc

def read_image(path):
    img = misc.imread(path)
    return img


def generate_test_batch(test_dir, batch_size):
    filelist = os.listdir(test_dir)
    while True:
        X_batch = []
        filenames = np.random.choice(filelist, size=batch_size, replace=False)
        for idx, img_fn in enumerate(filenames):
            img_path = os.path.join(test_dir, img_fn)
            x = read_image(img_path)
            X_batch.append(x)
        X = np.asarray(X_batch, dtype=np.float32)
        yield X