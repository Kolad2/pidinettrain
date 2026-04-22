import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from rocknetmanager.tools.shape_load import shape_load
import pandas as pd
from rocknetmanager.tools.image_data import ImageData
from tqdm import tqdm
import json


class DatasetPathList:
	def __init__(self, root):
		self.lst = pd.DataFrame({
			'images': [],
			'labels': []
		})
		self.root = root
		self.images = []
		self.labels = []

	def add(self, image, label):
		self.lst.loc[len(self.lst)] = [image, label]
		self.images.append(image)
		self.labels.append(label)

	def save(self):
		lst = {
			'images': self.images,
			'labels': self.labels
		}
		lst = pd.DataFrame(lst)
		lst.to_csv(str(self.root / "train.lst"), sep='\t', index=False, header=False)

	def __len__(self):
		return len(self.images)

	@classmethod
	def load(cls, load_path):
		lst = pd.read_csv(str(load_path), sep='\t', header=None)