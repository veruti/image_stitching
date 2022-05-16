from pathlib import Path

import cv2 as cv

from src.core.settings import settings


def read_images(images_dir: str):
    images_dir = Path(images_dir)
    image_paths = list(map(str, images_dir.glob(f"*.{settings.ALLOWED_IMAGES_TYPE}")))

    # Sort image path to get right images order
    image_paths = sorted(image_paths)
    return [cv.imread(image_path, flags=cv.IMREAD_COLOR) for image_path in image_paths]
