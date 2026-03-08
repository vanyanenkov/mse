# Task 1.2.5

import cv2
from pathlib import Path
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_images(dataset_path: str) -> List[str]:
    bad_files: List[str] = []
    path = Path(dataset_path)

    if not path.is_dir():
        raise ValueError(f"Путь '{dataset_path}' не является директорией или не существует.")

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    for file_path in path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            try:
                img = cv2.imread(str(file_path))
                if img is None:
                    raise ValueError("Изображение не удалось прочитать (возможно, повреждено).")
                if img.shape[0] == 0 or img.shape[1] == 0:
                    raise ValueError("Изображение имеет нулевые размеры.")
                logger.info(f"Файл '{file_path}' валиден.")
            except Exception as e:
                logger.warning(f"Невалидный файл '{file_path}': {str(e)}")
                bad_files.append(str(file_path))

    if bad_files:
        logger.error(
            f"Найдено {len(bad_files)} невалидных файлов. Рекомендуется удалить или исправить перед обучением YOLO.")

    return bad_files


if __name__ == "__main__":
    test_path = "app/core/datasets/coco8/images"
    invalid = validate_images(test_path)
    if invalid:
        print("Проблемные фалы:")
        for f in invalid:
            print(f)
    else:
        print("Изображения соотвествуют формату.")
