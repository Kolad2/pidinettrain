import cv2
from pathlib import Path
from storage_manager import Storage
from storage_manager.tools import save_shplines

import numpy as np

def main():
	thin_edges_path = Path("D:/PycharmProjects/pidinettrain/test_images/test_01_pink_thin_edges.png")
	#thin_edges_path = Path("test_images/test_01_thin_edges.png")
	storage = Storage.from_thin_edges_path(thin_edges_path)
	image_edges = storage.load_thin_edges()
	from extract_lines import extract_lines
	polylines = extract_lines(image_edges)

	save_shplines("D:/PycharmProjects/pidinettrain/test_images/test", polylines)

if __name__ == "__main__":
	main()
