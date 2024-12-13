import cv2
from pathlib import Path
from rocknetmanager.tools.vectorize import vectorize

def main():
	root_path = Path("D:/1.ToSaver/profileimages/photo_database_complited")
	image_folder = root_path / "IMGP6751"
	edges = load_thin_edges(image_folder)
	edges = edges[:,:,0]
	polylines = vectorize(edges, image_folder / "areas")




def load_thin_edges(image_folder: Path):
	edge_path = image_folder.name + "_edge_thin"
	edge_path = image_folder / str(edge_path)
	edge_path = edge_path.with_suffix(".png")
	return cv2.imread(edge_path)


if __name__ == "__main__":
	main()
