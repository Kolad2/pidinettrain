import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pygradskeleton import grayscale_skeletonize
from storage_manager import Storage


def main():
	# edges_path = Path("../test_images/test_01_edges.png")
	folder_path = Path("/media/koladik/HardDisk/segment_picture/20180811_191241")
	storage = Storage.from_folder_path(folder_path)
	# storage = Storage.from_edges_path(edges_path)
	edges = storage.load_edges()
	edges[edges < 125] = 0
	edges_thin = grayscale_skeletonize(edges, lam=20)
	storage.save_thin_edges(edges_thin)


if __name__ == "__main__":
	main()
