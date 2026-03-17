import cv2
import numpy as np
from pathlib import Path
import shapefile

def get_image_from_folder(image_folder: Path, size=None):
	image_path = image_folder / image_folder.stem
	image_path = image_path.with_suffix(".tif")
	image = get_image_from_path(image_path, size)
	return image

def get_image_from_path(image_path: Path, size=None):
	image = cv2.imread(str(image_path))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = image.astype(np.uint8)
	image = image if size is None else cv2.resize(image, size)
	return image

def save_edges(edges, image_folder: Path):
	edges_path = image_folder.name + "_edge"
	edges_path = image_folder / str(edges_path)
	edges_path = edges_path.with_suffix(".png")
	save_edges_path(edges, edges_path)

def save_edges_path(edges, edges_path: Path):
	image = ((edges / np.max(edges)) * 255).astype(np.uint8)
	image = cv2.merge((image, image, image))
	cv2.imwrite(str(edges_path), image)

def save_shplines(folder: Path, polylines, flip_y: bool = True):
	"""
	folder:
		Папка назначения. Внутри будут созданы:
		folder / (folder.stem + ".shp/.shx/.dbf")
	polylines:
		Итерируемый набор линий, каждая линия -- массив shape (N, 2)
	flip_y:
		Инвертировать ось Y перед записью
	"""
	folder = Path(folder)
	folder.mkdir(parents=True, exist_ok=True)

	shp_path = folder / f"{folder.stem}.shp"

	with shapefile.Writer(str(shp_path), shapeType=shapefile.POLYLINE, encoding="utf-8") as shp:
		shp.autoBalance = 1
		shp.field("ID", "N")
		shp.field("NAME", "C", size=64)

		for i, polyline in enumerate(polylines):
			pts = np.asarray(polyline, dtype=np.float64)

			# пропускаем мусор
			if pts.ndim != 2 or pts.shape[1] != 2:
				continue
			if len(pts) < 2:
				continue

			# удалим подряд идущие одинаковые точки
			if len(pts) > 1:
				keep = np.ones(len(pts), dtype=bool)
				keep[1:] = np.any(pts[1:] != pts[:-1], axis=1)
				pts = pts[keep]

			if len(pts) < 2:
				continue

			if flip_y:
				pts = pts * np.array([1.0, -1.0])

			shp.line([pts.tolist()])
			shp.record(i, f"Polyline_{i}")