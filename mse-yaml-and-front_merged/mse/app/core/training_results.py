"""
from ultralytics import YOLO
from pathlib import Path
import yaml


def create_test_dataset_config():

    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)

    dataset_config = {
        'path': './coco8',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                  5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light'}
    }

    config_path = config_dir / "test_dataset.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    return str(config_path)


def test_training():

    print("обучение YOLO...")

    try:
        model = YOLO("yolov8n.pt")
        print("ОК")

        dataset_path = create_test_dataset_config()


        results = model.train(
            data=dataset_path,
            epochs=1,
            imgsz=160,
            batch=4,
            device='cpu',
            verbose=True,
            project='runs/detect',
            name='test_train',
            exist_ok=True,
        )

        print("\nОбучение завершено")
        print(f"Результаты : runs/detect/test_train")

        print("\nРезультаты:")
        print(f"   - Метрики: {results}")

        return True

    except Exception as e:
        print(f"\Ошибка при обучении: {e}")
        return False


def test_prediction():


    try:
        model_path = "runs/detect/test_train/weights/best.pt"

        if not Path(model_path).exists():
            model = YOLO("yolov8n.pt")
        else:
            model = YOLO(model_path)


        import numpy as np
        from PIL import Image

        test_image = np.zeros((640, 640, 3), dtype=np.uint8)

        results = model(test_image)

        print("Предсказание выполнено")
        return True

    except Exception as e:
        print(f"ERROR {e}")
        return False


if __name__ == "__main__":
    success = test_training()
    if success:
        test_prediction()
    print("Тестирование завершено")

"""

'''
#Task 2.2.4

from ultralytics import YOLO
from pathlib import Path
from app.core.callbacks import create_early_stopping_callback
from app.core.yaml_generator import generate_coco8_yaml


def test_training_with_callback():

    try:
        model = YOLO("yolov8n.pt")
        print("Модель загружена")
        dataset_path = Path("coco8.yaml")
        if not dataset_path.exists():
            print("Создаем тестовый конфиг датасета")
            dataset_path = Path(generate_coco8_yaml())
        print(f"Датасет: {dataset_path}")

        # коллбек ранней остановки
        early_stopping = create_early_stopping_callback(
            patience=3,
            min_delta=0.15,
            min_epochs=5
        )

        model.add_callback('on_fit_epoch_end', early_stopping)

        results = model.train(
            data=str(dataset_path),
            epochs=20,
            imgsz=160,
            batch=8,
            device='cpu',
            project='runs/detect',
            name='test_early_stopping',
            exist_ok=True,
            verbose=True,
            plots=True,
        )
        print("Обучение завершено")

        print("\nРезультаты в: runs/detect/test_early_stopping")

        # Проверяем, сработал ли early stopping
        import csv
        csv_path = Path("runs/detect/test_early_stopping/results.csv")
        if csv_path.exists():
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                last_epoch = int(rows[-1][0]) if len(rows) > 1 else 0
                print(f"обучено эпох: {last_epoch + 1}/20")
                if last_epoch + 1 < 20:
                    print("early stopping")

        return True

    except Exception as e:
        print(f"\nОшибка: {e}")
        return False


def test_prediction():
    try:
        model_path = "runs/detect/test_early_stopping/weights/best.pt"

        if Path(model_path).exists():
            model = YOLO(model_path)
        else:
            print("Обученная модель не найдена, используем yolov8n.pt")
            model = YOLO("yolov8n.pt")

        import numpy as np
        from PIL import Image

        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(test_image)

        print(f"найдено {len(results[0].boxes)} объектов")
        return True

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


if __name__ == "__main__":
    success = test_training_with_callback()
    if success:
        test_prediction()
'''


import csv
from pathlib import Path


def read_latest_epoch_and_box_loss(training_dir: str | Path) -> dict[str, int | float]:
    """
    Ищет results.csv внутри training_dir, читает последнюю непустую строку
    и возвращает словарь с epoch и train/box_loss в числовом формате
    """

    base_path = Path(training_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"Training directory does not exist: {base_path}")

    if not base_path.is_dir():
        raise NotADirectoryError(f"Expected directory path, got file: {base_path}")

    results_path = _find_results_csv(base_path)

    with results_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(f"results.csv is empty or has no header: {results_path}")

        last_row: dict[str, str] | None = None

        for row in reader:
            normalized_row = {
                str(key).strip(): value.strip() if isinstance(value, str) else value
                for key, value in row.items()
                if key is not None
            }

            if _is_empty_row(normalized_row):
                continue

            last_row = normalized_row

        if last_row is None:
            raise ValueError(f"results.csv has no data rows: {results_path}")

    epoch_raw = _get_required_value(last_row, "epoch")
    box_loss_raw = _get_required_value(last_row, "train/box_loss")

    try:
        epoch = int(float(epoch_raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid epoch value: {epoch_raw}") from exc

    try:
        box_loss = float(box_loss_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid train/box_loss value: {box_loss_raw}") from exc

    return {
        "epoch": epoch,
        "train/box_loss": box_loss,
    }


def _find_results_csv(base_path: Path) -> Path:
    direct_path = base_path / "results.csv"
    if direct_path.is_file():
        return direct_path

    matches = sorted(base_path.rglob("results.csv"))

    if not matches:
        raise FileNotFoundError(f"results.csv not found in directory: {base_path}")

    return matches[0]


def _get_required_value(row: dict[str, str], key: str) -> str:
    if key not in row:
        available_columns = ", ".join(row.keys())
        raise KeyError(f"Column '{key}' not found. Available columns: {available_columns}")
    return row[key]


def _is_empty_row(row: dict[str, str]) -> bool:
    return all(value in ("", None) for value in row.values())