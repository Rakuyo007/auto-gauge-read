from typing import List

from ultralytics import YOLO
import cv2
from PIL import Image
from ocr import numbers_ocr
import math

def get_center_point(box):
    """计算矩形框的中心点"""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy

def get_angle(x, y):
    """
        计算点 (x, y) 相对于原点 (0, 0) 的角度。
        直角坐标系正方向为X轴向右Y轴向下，满足左手定则，极坐标系原点同直角坐标系，极轴为Y轴正方向。
    """
    return (math.degrees(-math.atan2(-y, x)) + 270) % 360

def interpolate_value(base_point_box: dict, confident_numbers: List[dict], confident_tip: dict) -> float:
    reference_points = []
    base_point = get_center_point(base_point_box)
    for confident_number in confident_numbers:
        box = confident_number['bbox']
        cx, cy = get_center_point(box)
        cx -= base_point[0]
        cy -= base_point[1]
        reference_points.append({
            "coo": (cx, cy),
            "angle": get_angle(cx, cy),
            "value": confident_number['ocr_result'][0]['content']
        })
        print("Reference Point:", reference_points[-1])
    tip_point_coo = get_center_point(confident_tip['bbox'])
    tip_point_angle = get_angle(tip_point_coo[0] - base_point[0], tip_point_coo[1] - base_point[1])
    print("Tip Point:", tip_point_angle)
    left = max((pt for pt in reference_points if pt['angle'] <= tip_point_angle), key=lambda x: x['angle'], default=None)
    right = min((pt for pt in reference_points if pt['angle'] >= tip_point_angle), key=lambda x: x['angle'], default=None)
    if left and right and left['angle'] != right['angle']:
        aL, vL = left['angle'], float(left['value'])
        aR, vR = right['angle'], float(right['value'])
        aT = tip_point_angle
        value_tip = vL + (vR - vL) * (aT - aL) / (aR - aL)
    else:
        value_tip = None  # 无法插值

    return value_tip

if __name__ == '__main__':
    image_path = "./test_imgs/3.jpg"
    model = YOLO("/Users/well/Documents/PyCharmProjects/yolo-model-train/runs/meter2/weights/best.pt")
    results = model.predict(source=image_path, show=False, save=True, imgsz=640, conf=0.3)
    preds = []
    for result in results:
        boxes = result.boxes  # 检测框对象
        names = model.names  # 类别ID到名称的映射
        for box in boxes:
            cls_id = int(box.cls.item())  # 类别ID
            conf = float(box.conf.item())  # 置信度
            xyxy = box.xyxy.cpu().numpy().tolist()[0]  # 坐标（左上右下）
            x1, y1, x2, y2 = map(int, xyxy)
            pred = {
                "name": names[cls_id],
                "bbox": [round(x, 2) for x in xyxy],  # [x1, y1, x2, y2]
                "confidence": round(conf, 4)
            }
            if pred['name'] == 'number':
                pil_image = Image.open(image_path).convert("RGB")
                cropped_img = pil_image.crop((x1, y1, x2, y2))
                ocr_result = numbers_ocr(cropped_img)
                pred["ocr_result"] = ocr_result
            preds.append(pred)
    confident_numbers = []
    for pred in preds:
        print(pred)
        if pred['name'] == 'number' and pred['ocr_result']:
            if pred['ocr_result'][0]['conf'] >= 0.95:
                confident_numbers.append(pred)
    print('-' * 16)
    confidnet_tip = max((pred for pred in preds if pred['name'] == 'tip'), key=lambda x: x['confidence'], default=None)
    base_point = next((pred for pred in preds if pred['name'] == 'base'), None)
    if confident_numbers and confidnet_tip:
        print("Confident Numbers:", confident_numbers)
        print("Confident Tip:", confidnet_tip)
        result = interpolate_value(base_point['bbox'], confident_numbers, confidnet_tip)
        print("Interpolated Value:", result)