import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pygradskeleton import grayscale_skeletonize
from storage_manager import Storage


def main():
	edges_path = Path("../test_images/test_01_edges.png")

	storage = Storage.from_edges_path(edges_path)
	edges = storage.load_edges()
	edges[edges < 125] = 0
	edges_thin = grayscale_skeletonize(edges, h=125)
	storage.save_thin_edges(edges_thin)


if __name__ == "__main__":
	main()
