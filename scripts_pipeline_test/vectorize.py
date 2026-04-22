import cv2
import numpy as np
from pathlib import Path
from storage_manager import Storage
from storage_manager.tools import save_shplines
from skelet_vectorize.extract_lines import extract_lines


def main():
	thin_edges_path = Path("D:/PycharmProjects/pidinettrain/test_images/test_01_pink_thin_edges.png")
	thin_edges_path = Path("../test_images/test_01_thin_edges.png")
	storage = Storage.from_thin_edges_path(thin_edges_path)
	image_edges = storage.load_thin_edges()
	polylines = extract_lines(image_edges)
	save_shplines("../test_images/test", polylines)


if __name__ == "__main__":
	main()
