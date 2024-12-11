import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pygradskeleton import grayscale_skeletonize


# image = cv2.imread("path/to/image")
# result_1 = grayscaleskelet(image, method="KIM")

def main():
	root_path = Path("D:/1.ToSaver/profileimages/photo_database_complited")
	image_folder = root_path / "IMGP3353-3355"
	edges = get_edges(image_folder)
	edges = edges[:, :, 0]
	edges[edges < 125] = 0
	edges_thin = grayscale_skeletonize(edges, h=125).astype(np.uint8)
	_, edges_thick = cv2.threshold(edges, 125, 255, cv2.THRESH_BINARY)

	edges_thin = cv2.merge((edges_thin, edges_thin, edges_thin))
	save_thin_edges(image_folder, edges_thin)

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


def get_edges(image_folder: Path):
	edge_path = image_folder.name + "_edge"
	edge_path = image_folder / str(edge_path)
	edge_path = edge_path.with_suffix(".png")
	return cv2.imread(str(edge_path))


def save_thin_edges(image_folder: Path, image: np.ndarray):
	edge_path = image_folder.name + "_edge_thin"
	edge_path = image_folder / str(edge_path)
	edge_path = edge_path.with_suffix(".png")
	cv2.imwrite(edge_path, image)


if __name__ == "__main__":
	main()
