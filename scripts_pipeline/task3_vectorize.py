import cv2
from pathlib import Path
from rocknetmanager.tools.vectorize import vectorize
from storage_manager import Storage


def main():
	thin_edges_path = Path("../test_images/test_01_thin_edges.png")
	storage = Storage.from_thin_edges_path(thin_edges_path)
	edges = storage.load_thin_edges()
	polylines = vectorize(edges, storage.folder_path / "edges")


if __name__ == "__main__":
	main()
