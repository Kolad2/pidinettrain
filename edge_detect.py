import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch.cuda

from rockedgesdetectors import ModelPiDiNet, ModelRCF, Cropper

models = {
	"pidinet_5": {
		"model": ModelPiDiNet,
		"checkpoint_path": "../models/pidinetmodels/table5_pidinet.pth"
	},
	"pidinet_7": {
		"model": ModelPiDiNet,
		"checkpoint_path": "../models/pidinetmodels/table7_pidinet.pth"
	},
	"pidinet_rock": {
		"model": ModelPiDiNet,
		"checkpoint_path": "save_models/checkpoint_000.pth"
	},
	"rcf": {
		"model": ModelRCF,
		"checkpoint_path": "../models/RCFcheckpoint_epoch12.pth"
	},
}


def main():
	model_name = "pidinet_rock"

	root_path = Path("D:/1.ToSaver/profileimages/photo_database_complited")
	image_folder = root_path / "IMGP6751"
	image = get_image(image_folder)
	#
	model = get_model(model_name)
	model = Cropper(model, crop=512, pad=64)

	edges = model(image)

	save_edges(edges, image_folder)



	fig = plt.figure(figsize=(14, 9))
	axs = [fig.add_subplot(1, 2, 1),fig.add_subplot(1, 2, 2)]
	axs[0].imshow(image)
	axs[1].imshow(edges)
	plt.show()


def get_model(name):
	return models[name]["model"](models[name]["checkpoint_path"])


def save_edges(edges, image_folder: Path):
	edge_path = image_folder.name + "_edge"
	edge_path = image_folder / str(edge_path)
	edge_path = edge_path.with_suffix(".png")
	image = ((edges/np.max(edges))*255).astype(np.uint8)
	image = cv2.merge((image, image, image))
	cv2.imwrite(str(edge_path), image)


def get_image(image_folder: Path, size=None):
	image_path = image_folder / image_folder.stem
	image_path = image_path.with_suffix(".png")
	image = cv2.imread(str(image_path))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = image.astype(np.uint8)
	image = image if size is None else cv2.resize(image, size)
	return image


if __name__ == "__main__":
	main()

