import os 
import numpy as np
import torch as t
from sklearn.metrics import confusion_matrix
from PIL import Image

from .csv_logger import CSVLogger

class SegmentationLogger():
    def __init__(self, directory, modelname):
        self.directory = directory
        self.score_log = CSVLogger(directory / 'scores.csv')

    def _confusion_stats(self,confusion):
        TP = np.diag(confusion)
        TP_FP = np.sum(confusion, axis=0) + 1e-8
        TP_FN = np.sum(confusion, axis=1) + 1e-8

        precision = TP / TP_FP
        recall = TP / TP_FN
        f_score = 2 * (precision * recall) / (precision + recall + 1e-8)

        return precision, recall, f_score

    def measure_score(self, label_1d, inferred_1d):
        confusion = confusion_matrix(label_1d, inferred_1d)
        precision, recall, f_score = self._confusion_stats(confusion)
        self.confusion = confusion / np.sum(confusion, axis=1, keepdims=True)
        score_dict = {}
        for i in range(precision.size):
            score_dict[f'precision class {i}'] = precision[i]
            score_dict[f'recall class {i}'] = recall[i]
            score_dict[f'f_score class {i}'] = f_score[i]
        score_dict[f'f_score stop'] = np.min(f_score)
        return score_dict

    def save_state(self,model):
        t.save(
            model,
            (self.directory / f'model-{model.name}.pth').as_posix(),
        )
        np.savetxt(self.directory / 'confusion.csv', self.confusion, '%.3f', delimiter=', ')
        
    def save_images(self, list_2dstack, epoch, subdirectoryname='valid_images'):
        stacked = np.dstack(list_2dstack)
        subdirectory = self.directory / f'{subdirectoryname}'
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)
        for slice_id, arr2d in enumerate(stacked):
            Image.fromarray(arr2d).save(subdirectory / f'{slice_id}_{epoch}.png')

    def save_images_v2(self, label_2dlist, label_2dlist_inferred, epoch, subdirectoryname='valid_images'):
        subdirectory = self.directory / f'{subdirectoryname}'
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)
        for slice_id in range(len(label_2dlist)):
            arr2d = np.hstack([label_2dlist[slice_id], label_2dlist_inferred[slice_id]])
            Image.fromarray(arr2d).save(subdirectory / f'{slice_id}_{epoch}.png')
