import cv2
import numpy as np
from pathlib import Path


class StorageManager:
    IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
                      ".PNG", ".JPG", ".JPEG", ".BMP", ".TIF", ".TIFF")
    debug = True

    def __init__(self):
        self.base_name: str | None = None
        self.folder_path: Path | None = None
        self.image_path: Path | None = None

    def load_image(self, size=None):
        self.image_path = self._resolve_image_path()
        if self.debug:
            print("Found image file", self.image_path)
        image = cv2.imread(str(self.image_path))
        if image is None:
            raise FileNotFoundError(f"cv2.imread() failed for: {self.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.uint8)
        image = image if size is None else cv2.resize(image, size)
        return image

    def _resolve_image_path(self) -> Path | None:
        if self.image_path is not None:
            return self.image_path
        if self.folder_path is None:
            raise ValueError("folder_path is not set")
        if not self.base_name:
            raise ValueError("base_name is not set")
        self.image_path = self._find_image_file(self.folder_path, self.base_name)
        if self.image_path is None:
            raise FileNotFoundError(
                f"Image file not found in {self.folder_path} with name {self.base_name}")
        return self.image_path

    @classmethod
    def _find_image_file(cls, folder_path: Path, name) -> Path | None:
        for suf in cls.IMAGE_SUFFIXES:
            candidate = folder_path / f"{name}{suf}"
            if candidate.is_file():
                return candidate
        return None

    @classmethod
    def from_image_path(cls, image_path: Path):
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        manager = cls()
        manager.image_path = image_path
        manager.base_name = image_path.stem
        manager.folder_path = image_path.parent
        return manager

    @classmethod
    def from_folder_path(cls, folder_path: Path):
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Folder not found: {folder_path}")
        manager = cls()
        manager.folder_path = folder_path
        manager.base_name = folder_path.name
        return manager
