import numpy as np
from shapely.geometry import LineString
from skan import Skeleton
from tqdm import tqdm

def extract_lines(image_thin, simplify_tol=1):
    """
    Возвращает список линий, каждая линия -- np.ndarray shape (N, 2), dtype float64.

    Координаты ставятся в центр пикселя:
        x = col + 0.5
        y = row + 0.5

    simplify_tol:
        tolerance для shapely.simplify().
        None или <= 0 -> без упрощения.
    """
    sk = Skeleton(image_thin, keep_images=False)
    lines = []

    for i in tqdm(range(sk.n_paths)):
        rc = sk.path_coordinates(i)   # shape (N, 2): row, col

        if len(rc) < 2:
            continue

        rows = rc[:, 0].astype(np.float64)
        cols = rc[:, 1].astype(np.float64)

        # координаты центра пикселя
        xy = np.column_stack((cols + 0.5, rows + 0.5))

        if simplify_tol is not None and simplify_tol > 0:
            line = LineString(xy).simplify(simplify_tol)

            if line.is_empty:
                continue

            xy = np.asarray(line.coords, dtype=np.float64)

            if len(xy) < 2:
                continue

        lines.append(xy)

    return lines