import cv2
import numpy as np
from pathlib import Path

def get_image_from_folder(image_folder: Path, size=None):
	image_path = image_folder / image_folder.stem
	image_path = image_path.with_suffix(".tif")
	image = get_image_from_path(image_path, size)
	return image

def get_image_from_path(image_path: Path, size=None):
	image = cv2.imread(str(image_path))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = image.astype(np.uint8)
	image = image if size is None else cv2.resize(image, size)
	return image

def save_edges(edges, image_folder: Path):
	edges_path = image_folder.name + "_edge"
	edges_path = image_folder / str(edges_path)
	edges_path = edges_path.with_suffix(".png")
	save_edges_path(edges, edges_path)


def save_edges_path(edges, edges_path: Path):
	image = ((edges / np.max(edges)) * 255).astype(np.uint8)
	image = cv2.merge((image, image, image))
	cv2.imwrite(str(edges_path), image)