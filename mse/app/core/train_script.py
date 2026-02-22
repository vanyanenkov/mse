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
    success = test_training_with_callback()
    if success:
        test_prediction()
