import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch.cuda

from rockedgesdetectors import ModelPiDiNet, ModelRCF, Cropper
from storage_manager import save_edges, save_edges_path, get_image_from_path, get_image_from_folder
from storage_manager import Storage

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
		"checkpoint_path": "../save_models/checkpoint_000.pth"
	},
	"rcf": {
		"model": ModelRCF,
		"checkpoint_path": "../models/RCFcheckpoint_epoch12.pth"
	},
}

def main():
	model_name = "pidinet_rock"
	# image_path = Path("../test_images/test_01.png")

	image_path = Path("/media/koladik/HardDisk/segment_picture/20180811_191241")
	storage = Storage.from_folder_path(image_path)
	# storage = Storage.from_image_path(image_path)

	image = storage.load_image()
	model = get_model(model_name)
	model = Cropper(model, crop=512, pad=64)
	edges = model(image)
	storage.save_edges(edges)

def get_model(name):
	return models[name]["model"](models[name]["checkpoint_path"])

if __name__ == "__main__":
	main()