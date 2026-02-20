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