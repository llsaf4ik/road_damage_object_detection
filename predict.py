import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO


def draw_boxes(image, result, class_names, colors):
    """
    Рисует ограничивающие рамки на изображении.
    """
    output_image = image.copy()
    
    # 1. Рассчитываем адаптивную толщину линий и размер шрифта
    thickness = max(2, int(min(output_image.shape[:2]) / 500))
    font_scale = max(0.5, min(output_image.shape[:2]) / 1600)

    boxes = result.boxes.xyxy.cpu().numpy()
    labels = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()

    for box, label_id, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        color = colors[label_id]
        class_name = class_names.get(label_id, 'Unknown')
        label_text = f"{class_name}: {score:.2f}"

        # Рисуем полупрозрачный фон для текста
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Создаем отдельный слой для фона
        overlay = output_image.copy()
        cv2.rectangle(overlay, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        
        alpha = 0.6  
        cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(output_image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    return output_image


def main(args):
    """
    Основная функция для загрузки модели и обработки изображения.
    """
    model = YOLO(args.weights)
    class_names = model.names
    colors = np.random.uniform(0, 255, size=(len(class_names), 3))

    # Поиск изображений
    if os.path.isfile(args.input):
        image_path = args.input
    else:
        raise FileNotFoundError(f"Не удалось найти файл или директорию: {args.input}")

    os.makedirs(args.output, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Не удалось прочитать изображение: {image_path}")
        return 

    # Предсказание с использованием оптимальных conf и iou
    results = model.predict(image, imgsz=1024, conf=0.2, iou=0.45, verbose=False)
    result_image = draw_boxes(image.copy(), results[0], class_names, colors)

    # Сохранение результата
    save_path = os.path.join(args.output, os.path.basename(image_path))
    cv2.imwrite(save_path, result_image)
    print(f"Результат сохранен в: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скрипт для детекции повреждений на дорогах с помощью YOLOv8.")
    parser.add_argument('--weights', type=str, required=True, help="Путь к файлу с весами модели (.pt).")
    parser.add_argument('--input', type=str, required=True, help="Путь к входному изображению или папке с изображениями.")
    parser.add_argument('--output', type=str, default='results', help="Папка для сохранения результатов.")
    
    args = parser.parse_args()
    main(args)