# Task 1.2.5

import cv2
from pathlib import Path
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_images(dataset_path: str) -> List[str]:
    bad_files: List[str] = []
    path = Path(dataset_path)

    if not path.is_dir():
        raise ValueError(f"Путь '{dataset_path}' неверный")

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    for file_path in path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            try:
                img = cv2.imread(str(file_path))
                if img is None:
                    raise ValueError("Ошибка чтения изображения")
                if img.shape[0] == 0 or img.shape[1] == 0:
                    raise ValueError("Изображение имеет нулевые размеры")
                logger.info(f"Файл '{file_path}' корректен")
            except Exception as e:
                logger.warning(f"Некорректный файл '{file_path}': {str(e)}")
                bad_files.append(str(file_path))

    if bad_files:
        logger.error(
            f"Найдено {len(bad_files)} некорректных файлов")

    return bad_files


if __name__ == "__main__":
    #пример
    test_path = "/path/to/your/dataset"
    invalid = validate_images(test_path)
    if invalid:
        print("Некорректные файлы:")
        for f in invalid:
            print(f)
    else:
        print("Все изображения подходят")


'''
# app/core/image_validator.py
# Задача 1.2.5: Валидатор изображений для датасета YOLO

import cv2
from pathlib import Path
from typing import List, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Поддерживаемые расширения изображений (YOLO-compatible)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Минимальные требования к изображению
MIN_IMAGE_SIZE = 32  # YOLO обычно требует изображения не меньше 32x32


def validate_single_image(file_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Проверяет одно изображение на целостность.

    Returns:
        Tuple[Optional[str], Optional[str]]: (путь_к_файлу_если_битый, сообщение_об_ошибке)
    """
    try:
        # Проверка размера файла (быстрая предварительная проверка)
        if os.path.getsize(file_path) == 0:
            return str(file_path), "Файл пустой (0 байт)"

        # Чтение изображения через OpenCV
        img = cv2.imread(str(file_path))

        if img is None:
            return str(file_path), "Не удалось прочитать изображение (cv2.imread вернул None)"

        # Проверка минимальных размеров
        height, width = img.shape[:2]
        if height < MIN_IMAGE_SIZE or width < MIN_IMAGE_SIZE:
            return str(
                file_path), f"Изображение слишком маленькое: {width}x{height} (мин. {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE})"

        # Проверка количества каналов (должно быть 1 или 3)
        channels = img.shape[2] if len(img.shape) == 3 else 1
        if channels not in [1, 3]:
            return str(file_path), f"Некорректное количество каналов: {channels}"

        # Дополнительная проверка: можно ли конвертировать в RGB (как это делает YOLO)
        try:
            if channels == 1:
                cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            return str(file_path), f"Ошибка конвертации цветового пространства: {str(e)}"

        return None, None  # Файл валиден

    except Exception as e:
        return str(file_path), f"Неожиданная ошибка: {str(e)}"


def validate_images(
        dataset_path: str,
        max_workers: int = 4,
        stop_on_first_error: bool = False
) -> List[str]:
    """
    Рекурсивно проверяет все изображения в датасете на расширение и целостность.

    Args:
        dataset_path: Абсолютный путь к директории датасета
        max_workers: Количество потоков для параллельной проверки
        stop_on_first_error: Останавливаться ли при первой ошибке

    Returns:
        List[str]: Список путей к невалидным файлам

    Raises:
        ValueError: Если путь некорректен или не является директорией
    """
    bad_files: List[str] = []
    path = Path(dataset_path)

    # Валидация входных параметров
    if not path.exists():
        raise ValueError(f"Путь '{dataset_path}' не существует.")
    if not path.is_dir():
        raise ValueError(f"Путь '{dataset_path}' не является директорией.")

    # Сбор всех файлов изображений
    image_files = []
    for file_path in path.rglob('*'):
        if (file_path.is_file() and
                file_path.suffix.lower() in IMAGE_EXTENSIONS and
                'labels' not in file_path.parts):  # Пропускаем папки с метками
            image_files.append(file_path)

    logger.info(f"Найдено {len(image_files)} изображений для проверки в {dataset_path}")

    # Параллельная проверка файлов
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(validate_single_image, f): f for f in image_files}

        for i, future in enumerate(as_completed(future_to_file), 1):
            bad_file, error_msg = future.result()

            if bad_file:
                bad_files.append(bad_file)
                logger.warning(f"[{i}/{len(image_files)}] НЕВАЛИДНЫЙ: {bad_file} - {error_msg}")

                if stop_on_first_error:
                    logger.error("Остановка по первому битому файлу")
                    break
            else:
                if i % 100 == 0:  # Логируем прогресс каждые 100 файлов
                    logger.info(f"Прогресс: {i}/{len(image_files)} файлов проверено")

    # Итоговый отчет
    if bad_files:
        error_summary = (
            f"\n{'=' * 60}\n"
            f"⚠️  ОБНАРУЖЕНЫ ПРОБЛЕМЫ: {len(bad_files)} из {len(image_files)} файлов повреждены\n"
            f"💡 Рекомендация: Запустите скрипт очистки или удалите проблемные файлы\n"
            f"📋 Список сохранен в переменной bad_files\n"
            f"{'=' * 60}"
        )
        logger.error(error_summary)
    else:
        success_message = (
            f"\n{'=' * 60}\n"
            f"✅ ВСЕ ИЗОБРАЖЕНИЯ ВАЛИДНЫ: {len(image_files)} файлов успешно проверено\n"
            f"📁 Датасет готов к обучению YOLO\n"
            f"{'=' * 60}"
        )
        logger.info(success_message)

    return bad_files


def cleanup_bad_files(bad_files: List[str], dry_run: bool = True) -> None:
    """
    Удаляет или перемещает битые файлы.

    Args:
        bad_files: Список путей к битым файлам
        dry_run: Если True, только показывает, что будет удалено
    """
    if dry_run:
        logger.info(f"Симуляция удаления {len(bad_files)} файлов:")
        for f in bad_files[:10]:  # Показываем первые 10
            logger.info(f"  Будет удален: {f}")
        if len(bad_files) > 10:
            logger.info(f"  ... и еще {len(bad_files) - 10} файлов")
    else:
        for f in bad_files:
            try:
                os.remove(f)
                logger.info(f"Удален: {f}")
            except Exception as e:
                logger.error(f"Не удалось удалить {f}: {e}")


# Пример использования
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        # Путь по умолчанию для теста
        #test_path = "app/core/datasets/coco8"
        test_path = "C:/Users/User/PycharmProjects/mse/app/core/datasets/football"

    print(f"\n🔍 Проверка датасета: {test_path}\n")

    try:
        # Основная проверка
        invalid_files = validate_images(
            dataset_path=test_path,
            max_workers=16,  # Используем 8 потоков для скорости
            stop_on_first_error=False
        )

        # Опционально: показать детали и удалить битые файлы
        if invalid_files:
            print(f"\nНайдено {len(invalid_files)} битых файлов")

            # Показать первые 20 битых файлов
            for f in invalid_files[:20]:
                print(f"  - {f}")

            # Опция удаления (по умолчанию dry_run=True для безопасности)
            response = input("\nУдалить битые файлы? (yes/no/dry): ").lower()
            if response == 'yes':
                cleanup_bad_files(invalid_files, dry_run=False)
            elif response == 'dry':
                cleanup_bad_files(invalid_files, dry_run=True)
            else:
                print("Файлы не удалены. Проверьте список и удалите вручную.")
        else:
            print("\n✅ Датасет чист! Можно запускать обучение YOLO.")

    except Exception as e:
        logger.error(f"Критическая ошибка при валидации: {e}")
        sys.exit(1)

'''