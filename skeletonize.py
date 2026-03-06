import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pygradskeleton import grayscale_skeletonize
from storage_manager import Storage

# image = cv2.imread("path/to/image")
# result_1 = grayscaleskelet(image, method="KIM")

def main():
	root_path = Path("/media/koladik/HardDisk/Image")
	image_folder = root_path / "ESRI_cut02_5m"

	edges_path = Path("test_images/test_01_edges.png")
	storage = Storage.from_edges_path(edges_path)
	edges = storage.load_edges()

	#edges[edges < 125] = 0
	edges_thin = grayscale_skeletonize(edges, h=125).astype(np.uint8)
	_, edges_thick = cv2.threshold(edges, 125, 255, cv2.THRESH_BINARY)

	storage.save_thin_edges(edges_thin)

	fig = plt.figure(figsize=(8, 6))
	axs = [
		fig.add_subplot(2, 2, 1),
		fig.add_subplot(2, 2, 2),
		fig.add_subplot(2, 2, 3)
	]
	axs[0].imshow(edges)
	axs[1].imshow(edges_thin)
	axs[2].imshow(edges_thick)
	#
	axs[1].sharex(axs[0])
	axs[1].sharey(axs[0])
	#
	axs[2].sharex(axs[1])
	axs[2].sharey(axs[1])

	plt.show()



if __name__ == "__main__":
	main()
