from itertools import product
from ultralytics import YOLO
from pathlib import Path

class GridSearch:
    def __init__(self, model_path, parametrs_dict, save_dir="grid_search_results", validation_metric="map50_95"):
        self.model_path = model_path
        self.parametrs_dict = parametrs_dict
        self.best_score = None
        self.best_params = None
        self.best_model = None
        self.path_to_best_model = None
        self.save_dir = Path(save_dir)
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
        box_metrics = ['map50', 'map75', 'map50_95', 'precision','recall','ap_class_index','class_aps','f1','fitness']

        if validation_metric not in box_metrics:
            print("Неизвестная метрика для сравнения моделей! Используется метрика по умолчанию map50_95")
            self.validation_metric = "map50_95"
        else:
            self.validation_metric = validation_metric

    def train(self, data):
        paramet_combinations = self.product_dict()
        for i, params in enumerate(paramet_combinations):
            print(f"Обучение с параметрами: {params} и максимизацией метрики {self.validation_metric}")
            model = YOLO(self.model_path)


            params_with_path = params.copy()
            params_with_path['project'] = str(self.save_dir)
            params_with_path['name'] = f"exp_{i}"

            model.train(data=data, **params_with_path)

            current_model_path = self.save_dir / f"exp_{i}"

            metrics = model.val()
            box_metrics = {
                'map50': metrics.box.map50,          
                'map75': metrics.box.map75,       
                'map50_95': metrics.box.map,      
                'precision': metrics.box.p[0],    
                'recall': metrics.box.r[0],      
                'ap_class_index': metrics.box.ap_class_index, 
                'class_aps': metrics.box.ap,                  
                'f1': metrics.box.f1[0],            
                'fitness': metrics.box.fitness,     
            }
            if self.best_score is None or box_metrics[self.validation_metric] > self.best_score:
                 self.best_score = box_metrics[self.validation_metric]
                 self.best_params = params.copy()
                 self.best_model = model
                 self.path_to_best_model = current_model_path

        print(f"Лучшие параметры: {self.best_params}")
        print(f"Путь к наилучшей модели: {self.path_to_best_model}")
        return self.best_model
    
    def product_dict(self):
        if not self.parametrs_dict:
            return []
        
        unvalid_keys = []
        for key in self.parametrs_dict:
          if key not in self.valid_args:
            print(f"Неизвестный аргумент: {key}! Обучение выполняется без {key}")
            unvalid_keys.append(key)

        for key in unvalid_keys:
          del self.parametrs_dict[key]

        keys = self.parametrs_dict.keys()
        combinations = [dict(zip(keys, p)) for p in product(*self.parametrs_dict.values())]

        return combinations



if __name__ == "__main__":
    model_path = "yolov8n.pt"
    data_path = "coco8.yaml"
    params = {
        "epochs": [1, 2],
        "XYZ": [1, 2],
        "iou": [0.3, 0.5]
    }
    
    searcher = GridSearch(model_path, params)
    best_model= searcher.train(data=data_path)
    
    print(f"\nОбучение завершено!")