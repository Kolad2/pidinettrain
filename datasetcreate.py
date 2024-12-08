import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from rocknetmanager.tools.shape_load import shape_load
import pandas as pd
from rocknetmanager.tools.image_data import ImageData
from tqdm import tqdm
import json


def main():
	with open("./config.json") as file:
		config = json.load(file)
		image_folders = [Path(path) for path in config["databases"]]
		dataset_root = Path(config["train_folder"])

	lst = DatasetPathList(dataset_root)

	images_folder = image_folders[0]
	process_folder(images_folder, lst)

	#images_folder = image_folders[1]
	#process_folder(images_folder, lst)

	images_folder = image_folders[2]
	process_folder(images_folder, lst)

	lst.save()


def process_folder(images_folder, lst):
	list_folder_images = [path for path in images_folder.iterdir()]
	for image_folder in list_folder_images:
		print("Processing:", image_folder.stem)
		data, bbox = ImageData.load(image_folder)
		save_path = lst.root / image_folder.stem
		save_path.mkdir(parents=False, exist_ok=True)
		process_image(data, bbox, lst, save_path=save_path)


def crop_save(data, save_path, lst):
	if data.is_accessible():
		image_path, label_path = data.save(
			root=save_path,
			name=str(len(lst))
		)
		image_path = image_path.relative_to(lst.root)
		label_path = label_path.relative_to(lst.root)
		lst.add(str(image_path), str(label_path))


def process_image(data: ImageData, bbox, lst, save_path=None, cropshift=(512, 512), cropres=(512, 512)):
	save_path = lst.root if save_path is None else save_path
	list_x = np.arange(bbox[0], bbox[2], cropshift[0])
	list_y = np.arange(bbox[1], bbox[3], cropshift[1])
	#
	paths = {
		"rotate_0": save_path / "rotate_0",
		"rotate_90": save_path / "rotate_90",
		"rotate_180": save_path / "rotate_180",
		"rotate_270": save_path / "rotate_270"
	}
	for key in paths:
		paths[key].mkdir(parents=False, exist_ok=True)
	coordinates = [(x, y) for y in list_y for x in list_x]
	for (x, y) in tqdm(coordinates):
		cropped_data = data.crop_image(x, y, cropres[0], cropres[1])
		for key in paths:
			crop_save(cropped_data, paths[key], lst)
			cropped_data.rotate()


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




main()

