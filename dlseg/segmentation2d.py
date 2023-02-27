import numpy as np
import torch as t
import torch.nn as nn
from torch.optim import RAdam
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as tvf
from random import random, choice
from tqdm import trange
from pathlib import Path
from collections import defaultdict
import os

from .resunet import ResUNet
from .sample_augment import Sampler_Augmenter
from .segmentation_logger import SegmentationLogger

class Segmentation2D:
    def __init__(
        self,
        directory,
        device='cpu',
    ):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.directory = Path(directory)
        self.device = device

    @property
    def tile_size(self):
        return self.model.tile_size

    @property
    def tile_overlap(self):
        return self.model.tile_size // 8

    @property
    def pad_size(self):
        return self.model.tile_size

    def new_model(self, name, network, num_classes, tile_size):
        self.model = ResUNet(name, tile_size, 1, num_classes, residuals='res' in network)

    def load_model(self, model_name):
        self.model = t.load((self.directory / f'model-{model_name}.pth').as_posix())

    def _to_tensor(self, arr2d, is_label=False):# pad, normalize, convert to torch
        if arr2d.ndim != 2:
            raise ValueError('Array must have exactly two dimensions')
        arr2d = np.pad(arr2d, self.pad_size, mode='symmetric')
        tsr = t.from_numpy(arr2d)
        if is_label:
            tsr = tsr.long()
        else:
            tsr = tsr.float()
            tsr = (tsr - tsr.mean()) / (tsr.std() + 1e-8)
        return tsr

    def _get_label_weights(self, label_2dlst):
        flattened = t.cat([ele.flatten() for ele in label_2dlst])
        return 1 - t.bincount(flattened) / flattened.numel()

    def _iter_tile_batches(self, width, height, batch_size):
        tile_coords = []

        for x in range(0, width - self.tile_size, self.tile_size - self.tile_overlap):
            for y in range(0, height - self.tile_size, self.tile_size - self.tile_overlap):
                tile_coords.append((x, y))

        for i in range(0, len(tile_coords), batch_size):
            yield tile_coords[i : i + batch_size]

    def infer(self, gray_tsr_2d, batch_size=4):
        width, height = gray_tsr_2d.size(-2), gray_tsr_2d.size(-1)
        margin = self.tile_overlap // 2

        result_tsr = t.zeros((width, height), dtype=t.uint8)

        was_training = self.model.training
        self.model.eval()

        for batch_coords in self._iter_tile_batches(width, height, batch_size):
            batch_in = []
            for x, y in batch_coords:
                batch_in.append(
                    gray_tsr_2d[None, None, x : x + self.tile_size, y : y + self.tile_size]
                )
            batch_in = t.cat(batch_in, dim=0).to(device=self.device)

            with t.no_grad():
                batch_out = t.argmax(
                    self.model(batch_in)[..., margin:-margin, margin:-margin], dim=1
                ).cpu()

            for (x, y), tile in zip(batch_coords, batch_out):
                result_tsr[
                    x + margin : x + self.tile_size - margin,
                    y + margin : y + self.tile_size - margin,
                ] = tile

        self.model.train(was_training)

        result_tsr = result_tsr[self.pad_size : -self.pad_size, self.pad_size : -self.pad_size]
        result_arr = result_tsr.detach().numpy()

        return result_arr

    def infer_2dlist(self, gray, batch_size=4, showprogress=False):
        N = len(gray)
        if showprogress:
            iter_range = trange(N)
        else:
            iter_range = range(N)
        inferred_tsr_2dlist = []
        for i in iter_range:
            gray2d = gray[i]

            if isinstance(gray2d, t.Tensor):
                gray_tsr_2d = gray2d# when gray is list of 2d tensor for inference of validation data
            else:
                gray_tsr_2d = self._to_tensor(gray2d)# when gray is 3d numpy array for inference of test data

            inferred_tsr_2d = self.infer(gray_tsr_2d, batch_size)
            inferred_tsr_2dlist.append(inferred_tsr_2d)
        return inferred_tsr_2dlist

    def train(
        self,
        num_epochs=30,
        steps_per_epoch=100,
        batch_size=4,
        learning_rate=3e-4,
        learning_rate_decay=0.9,
        f_score_change_min = 0.0001,
        MAX_PATIENCE = 10,
    ):


        class_weights = self._get_label_weights(self.train_label_tsr_2dlist)
        loss = nn.CrossEntropyLoss(class_weights).to(device=self.device)
        optimizer = RAdam(self.model.parameters(), learning_rate)
        scheduler = StepLR(optimizer, step_size=steps_per_epoch, gamma=learning_rate_decay)
        sampler_augmenter = Sampler_Augmenter(self.model.tile_size)
        
        logger = SegmentationLogger(self.directory, self.model.name)
        logger.score_log.start_log()

        self.model.to(device=self.device).train()
        train_tsrs = [
            t.stack([
                self.train_gray_tsr_2dlist[i], 
                self.train_label_tsr_2dlist[i]
            ], dim=0)[None, ...].to(device=self.device)
            for i in range(len(self.train_gray_tsr_2dlist))
        ] # shape: [1, 2, x_withpad, y_withpad]

        best_stop_f_score = 0
        patience = 0

        concatenate = lambda list2d: np.concatenate([ele.ravel() for ele in list2d]) # concatenate elements in list of 2d array into a long 1d arrayy

        for epoch in trange(num_epochs):
            for step in trange(steps_per_epoch, leave=False):
                optimizer.zero_grad()
                batch = sampler_augmenter.get_batch(train_tsrs,batch_size)
                output_tsr = self.model(batch[:, 0:1])# train

                loss_value = loss(output_tsr, batch[:, 1].long())
                loss_value.backward()
                optimizer.step()
                scheduler.step()

            valid_label_2dlist_inferred = self.infer_2dlist(self.valid_gray_tsr_2dlist, batch_size)
            
            # Checkpointing and Early Stopping
            score_dict = logger.measure_score(concatenate(self.valid_label_2dlist),concatenate(valid_label_2dlist_inferred))
            logger.score_log.log_dict(epoch, score_dict)
            if score_dict['f_score stop'] - best_stop_f_score > f_score_change_min:
                best_stop_f_score = score_dict['f_score stop']
                patience = 0
                logger.save_state(self.model)
                logger.save_images_v2(self.valid_label_2dlist, valid_label_2dlist_inferred, epoch)
            else:
                patience += 1
                if patience == MAX_PATIENCE:
                    break

        logger.score_log.end_log()

    def ndarray_to_tensor(self, train_gray_2dlist, train_label_2dlist, valid_gray_2dlist, valid_label_2dlist):
        self.train_gray_tsr_2dlist = [self._to_tensor(arr2d) for arr2d in train_gray_2dlist]
        self.train_label_tsr_2dlist = [self._to_tensor(arr2d, True) for arr2d in train_label_2dlist]
        
        self.valid_gray_tsr_2dlist = [self._to_tensor(arr2d) for arr2d in valid_gray_2dlist]
        self.valid_label_2dlist = valid_label_2dlist
