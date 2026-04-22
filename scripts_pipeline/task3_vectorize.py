import cv2
import numpy as np
from pathlib import Path
from storage_manager import Storage
from storage_manager.tools import save_shplines
from skelet_vectorize.extract_lines import extract_lines


def main():
	folder_path = Path("/media/koladik/HardDisk/segment_picture/20180811_191241")
	storage = Storage.from_folder_path(folder_path)

	image_edges = storage.load_thin_edges()

	polylines = extract_lines(image_edges)

	save_shplines("/media/koladik/HardDisk/segment_picture/20180811_191241/test", polylines)



if __name__ == "__main__":
	main()


