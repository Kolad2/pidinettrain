from sys import path_hooks

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from rocknetmanager.tools.shape_load import shape_load
import pandas as pd
from rocknetmanager import ImageData, DatasetPathList, ImageTiler
from tqdm import tqdm
import json


def main():
	# Входной lst с сырыми данными
	raw_lst_path = Path(r"D:\Data\Outcrops\dataset_raw.lst")

	# Корень сырых данных.
	# Если пути в raw_lst_path уже считаются относительно папки, где лежит dataset.lst,
	# можно оставить root_path=None.
	raw_root = raw_lst_path.parent

	# Куда сохраняем новый датасет
	dataset_root = Path(r"D:\Data")

	# Куда сохраняем новый lst
	output_lst_path = dataset_root / r"dataset.lst"

	cropres = (512, 512)
	cropshift = (337, 337)

	# Загружаем сырой список
	lstraw = DatasetPathList.load(
		lst_path=raw_lst_path,
		root_path=raw_root,
	)
	lst_selftrain = DatasetPathList.load(
		lst_path=Path(r"D:\Data\Outcrops\dataset_selftrain.lst"),
		root_path=raw_root,
	)

	with DatasetPathList(
		root_path=dataset_root,
		save_path=output_lst_path
	) as lst:
		# Обрабатываем
		process_lst(
			lstraw=lstraw,
			lst=lst,
			save_path=dataset_root / "train_data",
			cropshift=cropshift,
			cropres=cropres,
		)

		process_lst(
			lstraw=lst_selftrain,
			lst=lst,
			save_path=dataset_root / "train_data",
			cropshift=cropshift,
			cropres=cropres,
		)


def process_lst(
	lstraw: DatasetPathList,
	lst: DatasetPathList,
	cropshift,
	cropres,
	save_path: Path | None = None,
):
	if lstraw.root is None:
		raise ValueError("У lstraw.root должен быть задан root_path")

	if lst.root is None:
		raise ValueError("У lst.root должен быть задан root_path")

	save_path = Path(lst.root if save_path is None else save_path).resolve()
	lst_root = Path(lst.root).resolve()

	try:
		save_path.relative_to(lst_root)
	except ValueError:
		raise ValueError(
			f"save_path={save_path} должен лежать внутри lst.root={lst_root}"
		)

	save_path.mkdir(parents=True, exist_ok=True)

	tiler = ImageTiler(
		tile_resolution=cropres,
		cropshift=cropshift,
		lst=lst,
		rotation=True,
	)

	pbar = tqdm(
		lstraw.lst.iterrows(),
		total=len(lstraw),
		desc="Processing raw dataset",
		position=0,
		leave=True,
		dynamic_ncols=True,
	)

	for idx, row in pbar:
		path_image = resolve_lst_path(lstraw, row["images"])
		path_labels = resolve_lst_path(lstraw, row["labels"])
		path_mask = resolve_lst_path(lstraw, row.get("masks", None))

		if path_image is None:
			raise ValueError(f"В строке {idx} не задан путь к image")

		if path_labels is None:
			raise ValueError(f"В строке {idx} не задан путь к label")

		pbar.set_postfix_str(path_image.stem)

		data, bbox = ImageData.load(
			path_image=path_image,
			path_labels=path_labels,
			path_mask=path_mask,
			thickness=1
		)

		sample_name = f"{idx:06d}_{path_image.stem}"

		sample_save_path = save_path / sample_name
		sample_save_path.mkdir(parents=True, exist_ok=True)

		tiler.tile_image(
			data=data,
			bbox=None,
			name_image=sample_name,
			save_path=sample_save_path,
		)

def resolve_lst_path(dataset: DatasetPathList, value) -> Path | None:
	if value is None or pd.isna(value):
		return None

	path = Path(value)

	if path.is_absolute():
		return path

	if dataset.root is None:
		return path

	return dataset.root / path


if __name__ == "__main__":
	main()
