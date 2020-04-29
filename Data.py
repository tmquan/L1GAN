
#
import tensorpack.dataflow as df
from tensorpack.utils import get_rng
from tensorpack.utils.argtools import shape2d

import os
import cv2
import glob
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

class MultiLabelCXRDataset(df.RNGDataFlow):
    def __init__(self, folder, types=14, is_train='train', channel=3,
                 resize=None, debug=False, shuffle=False, pathology=None, 
                 fname='train.csv', balancing=None):

        self.version = "1.0.0"
        self.description = "Vinmec is a large dataset of chest X-rays\n",
        self.citation = "\n"
        self.folder = folder
        self.types = types
        self.is_train = is_train
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        if self.channel == 1:
            self.imread_mode = cv2.IMREAD_GRAYSCALE
        else:
            self.imread_mode = cv2.IMREAD_COLOR
        if resize is not None:
            resize = shape2d(resize)
        self.resize = resize
        self.debug = debug
        self.shuffle = shuffle
        self.csvfile = os.path.join(self.folder, fname)
        print(self.folder)
        # Read the csv
        self.df = pd.read_csv(self.csvfile)
        self.df.columns = self.df.columns.str.replace(' ', '_')
        self.df = self.df.infer_objects()
        
        self.pathology = pathology
        self.balancing = balancing
        if self.balancing == 'up':
            self.df_majority = self.df[self.df[self.pathology]==0]
            self.df_minority = self.df[self.df[self.pathology]==1]
            print(self.df_majority[self.pathology].value_counts())
            self.df_minority_upsampled = resample(self.df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=self.df_majority[self.pathology].value_counts()[0],    # to match majority class
                                     random_state=123) # reproducible results

            self.df_upsampled = pd.concat([self.df_majority, self.df_minority_upsampled])
            self.df = self.df_upsampled
    def reset_state(self):
        self.rng = get_rng(self)

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        indices = list(range(self.__len__()))
        if self.is_train == 'train':
            self.rng.shuffle(indices)

        for idx in indices:
            fpath = os.path.join(self.folder, 'data')
            fname = os.path.join(fpath, self.df.iloc[idx]['Images'])
            image = cv2.imread(fname, self.imread_mode)
            assert image is not None, fname
            # print('File {}, shape {}'.format(fname, image.shape))
            if self.channel == 3:
                image = image[:, :, ::-1]
            if self.resize is not None:
                image = cv2.resize(image, tuple(self.resize[::-1]))
            if self.channel == 1:
                image = image[:, :, np.newaxis]

            # Process the label
            if self.is_train == 'train' or self.is_train == 'valid':
                label = []
                if self.types == 6:
                    label.append(self.df.iloc[idx]['Airspace_Opacity'])
                    label.append(self.df.iloc[idx]['Cardiomegaly'])
                    label.append(self.df.iloc[idx]['Fracture'])
                    label.append(self.df.iloc[idx]['Lung_Lesion'])
                    label.append(self.df.iloc[idx]['Pleural_Effusion'])
                    label.append(self.df.iloc[idx]['Pneumothorax'])
                if self.types == 4:
                    label.append(self.df.iloc[idx]['Covid'])
                    label.append(self.df.iloc[idx]['Airspace_Opacity'])
                    label.append(self.df.iloc[idx]['Consolidation'])
                    label.append(self.df.iloc[idx]['Pneumonia'])
                elif self.types == 2:
                    assert self.pathology is not None
                    label.append(self.df.iloc[idx]['No_Finding'])
                    label.append(self.df.iloc[idx][self.pathology])
                else:
                    label.append(-1)

                # Try catch exception
                label = np.nan_to_num(label, copy=True, nan=0)
                label = np.array(label, dtype=np.float32)
                types = label.copy()
                yield [image, types]
            elif self.is_train == 'test':
                yield [image] 
            else:
                pass
