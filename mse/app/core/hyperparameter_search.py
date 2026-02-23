from itertools import product
import random
from typing import Dict, List, Any, Tuple
from ultralytics import YOLO
from pathlib import Path

def grid_search_params(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    param_grid_copy = {}
    for key in param_grid.keys():
        if param_grid[key] != []:
            param_grid_copy[key] = param_grid[key]

    keys = param_grid_copy.keys()
    values = param_grid_copy.values()
    combinations = [dict(zip(keys, combo)) for combo in product(*values)]
    return combinations


def random_search_params(param_ranges: Dict[str, Tuple], n_iter: int = 10) -> List[Dict[str, Any]]:
    if not param_ranges:
        return []

    combinations = []


    for _ in range(n_iter):
        combo = {}
        unique_flag = False
        generate_count = 0
        while not unique_flag and generate_count < 10:
            for key, params in param_ranges.items():
                if len(params) == 0:
                    continue

                if len(params) == 1:
                    combo[key] = params[0]
                else:
                    if isinstance(params[0], int) and isinstance(params[1], int):

                        combo[key] = random.randint(params[0], params[1])
                    elif isinstance(params[0], float) or isinstance(params[1], float):

                        combo[key] = random.uniform(params[0], params[1])

                    elif isinstance(params[0], str) or isinstance(params[1], str):
                        combo[key] = params[random.randint(0, len(params) - 1)]

                    else:
                        combo[key] = random.uniform(float(params[0]), float(params[1]))               
            
            generate_count += 1

            if combo not in combinations:
                combinations.append(combo)
                unique_flag = True

    return combinations

class GridSearch:

    def __init__(
            self,
            model_path: str,
            params_dict: Dict[str, List[Any]],
            save_dir: str = "runs/detect/grid_search",
            validation_metric: str = "map50_95"
    ):
        self.model_path = model_path
        self.params_dict = params_dict
        self.save_dir = Path(save_dir)
        self.validation_metric = validation_metric

        self.best_score = None
        self.best_params = None
        self.best_model = None
        self.path_to_best_model = None

        self.valid_args = [
            'epochs', 'time', 'patience', 'batch', 'imgsz',
            'save', 'save_period', 'cache', 'device', 'workers',
            'exist_ok', 'pretrained', 'optimizer', 'seed', 'deterministic',
            'single_cls', 'classes', 'rect', 'cos_lr', 'close_mosaic', 'resume',
            'amp', 'fraction', 'profile', 'freeze', 'lr0', 'lrf', 'momentum',
            'weight_decay', 'warmup_epochs', 'warmup_momentum', 'warmup_bias_lr',
            'box', 'cls', 'dfl', 'pose', 'kobj', 'nbs', 'overlap_mask',
            'mask_ratio', 'dropout', 'val', 'plots', 'multi_scale', 'verbose',
            'augment', 'agnostic_nms', 'retina_masks', 'max_det', 'half',
            'dnn', 'source', 'vid_stride', 'stream_buffer', 'visualize',
            'save_txt', 'save_conf', 'save_crop', 'show_labels', 'show_conf',
            'line_width', 'format', 'keras', 'optimize', 'int8', 'dynamic',
            'simplify', 'opset', 'workspace', 'nms', 'compile', 'conf', 'iou'
        ]

        self.available_metrics = [
            'map50', 'map75', 'map50_95', 'precision', 'recall', 'f1', 'fitness'
        ]

        if validation_metric not in self.available_metrics:
            print(f"Неизвестная метрика '{validation_metric}'. Используется 'map50_95'")
            self.validation_metric = "map50_95"

    def _filter_valid_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        valid_params = {}
        for key, value in params.items():
            if key in self.valid_args:
                valid_params[key] = value
            else:
                print(f"Предупреждение: параметр '{key}' недопустим и будет пропущен")

        return valid_params

    def _extract_metrics(self, metrics) -> Dict[str, float]:

        return {
            'map50': metrics.box.map50,
            'map75': metrics.box.map75,
            'map50_95': metrics.box.map,
            'precision': metrics.box.p[0] if len(metrics.box.p) > 0 else 0,
            'recall': metrics.box.r[0] if len(metrics.box.r) > 0 else 0,
            'f1': metrics.box.f1[0] if len(metrics.box.f1) > 0 else 0,
            'fitness': metrics.box.fitness,
        }

    def train(self, data: str) -> YOLO:

        param_combinations = grid_search_params(self.params_dict)

        print(f"Запуск обучения с {len(param_combinations)} комбинациями")
        print(f"Метрика для сравнения: {self.validation_metric}")

        for i, params in enumerate(param_combinations):
            print(f"Обучение {i + 1}/{len(param_combinations)}")
            print(f"Параметры: {params}")

            valid_params = self._filter_valid_params(params)
            model = YOLO(self.model_path)


            train_params = valid_params.copy()
            train_params['project'] = str(self.save_dir)
            train_params['name'] = f"exp_{i:03d}"
            train_params['exist_ok'] = True

            model.train(data=data, **train_params)
            metrics_result = model.val()
            metrics = self._extract_metrics(metrics_result)
            current_score = metrics.get(self.validation_metric, 0)

            print(f" {self.validation_metric}: {current_score:.4f}")

            if self.best_score is None or current_score > self.best_score:
                self.best_score = current_score
                self.best_params = params.copy()
                self.best_model = model
                self.path_to_best_model = self.save_dir / f"exp_{i:03d}"
                print(f"Новая лучшая модель Score: {self.best_score:.4f}")


        print(f"Лучшие параметры: {self.best_params}")
        print(f"Путь к лучшей модели: {self.path_to_best_model}")
        print(f"Лучший score ({self.validation_metric}): {self.best_score:.4f}")

        return self.best_model
