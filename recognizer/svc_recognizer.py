import os
import pickle
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn import svm
from sklearn.preprocessing import LabelEncoder

TRAINING_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 63
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pickle')


def _load_data():
    level_root = Path(TRAINING_DATA_PATH)
    level_dirs = [directory for directory in level_root.iterdir() if directory.is_dir()]
    level_classes = {level_dir.name: [class_dir for class_dir in level_dir.iterdir() if level_dir.is_dir()] for
                     level_dir in level_dirs if
                     level_dir.is_dir()}

    flat_data = {}
    targets = {}
    levels = []
    for level, classes in level_classes.items():
        class_data = []
        class_target = []
        for class_path in classes:
            for file in class_path.iterdir():
                img = Image.open(file)
                # img_resized = img.resize((IMAGE_HEIGHT, IMAGE_WIDTH))
                class_data.append(np.array(img).flatten())
                class_target.append(str(class_path).split('/')[-1])
        flat_data[level] = np.array(class_data)
        targets[level] = np.array(class_target)
        levels.append(level)
        # print('data shape: {}'.format(flat_data.shape))
    return flat_data, targets, levels


class Recognizer(object):
    def __init__(self):
        self.clfs = {}
        self.les = {}
        if os.path.isfile(MODEL_PATH):
            st = time.time()
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
                self.les = model['le']
                self.clfs = model['clf']
            print(time.time() - st)
        else:
            self.fit(save_model=True)

    def fit(self, save_model=False):
        x, y, levels = _load_data()
        for level in levels:
            x_in_level = x[level]
            y_in_level = y[level]
            le = LabelEncoder()
            clf = svm.SVC()
            y_encoded = le.fit_transform(y_in_level)
            clf.fit(x_in_level, y_encoded)
            self.clfs[level] = clf
            self.les[level] = le
        print('Fit Complete')

        if save_model:
            with open(MODEL_PATH, 'wb+') as f:
                model = {
                    'le': self.les,
                    'clf': self.clfs
                }
                pickle.dump(model, f)

    def produce(self, x, level):
        y_pred = self.clfs[level].predict(x)
        y_original = self.les[level].inverse_transform(y_pred)
        return y_original.tolist()

# if __name__ == '__main__':
#     recog = Recognizer()
