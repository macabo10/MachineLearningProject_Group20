import pandas
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import cv2
import os
from nltk import wordpunct_tokenize
import re
import Utils

class PosterModel(nn.Module):

    def __init__(self, train_dataloader, test_dataloader):
        super(PosterModel, self).__init__()
        self.hidden_size = Utils.HIDDEN_SIZE

        pass