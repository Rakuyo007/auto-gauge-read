from typing import List

from ultralytics import YOLO
import cv2
from PIL import Image
from ocr import numbers_ocr
import math

def get_center_point(box: list) -> tuple:
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

def interpolate_value(base_point_box: dict, confident_numbers: List[dict], confident_tip: dict, calibration_value: float) -> float:
    reference_points = []
    base_point = get_center_point(base_point_box)
    for confident_number in confident_numbers:
        box = confident_number['bbox']
        cx, cy = get_center_point(box)
        cx -= base_point[0]
        cy -= base_point[1]
        reference_points.append({
            "coo": (cx, cy),
            "angle_raw": get_angle(cx, cy),
            "angle": get_angle(cx, cy) + calibration_value,
            "value": confident_number['ocr_result'][0]['content']
        })
        print("Reference Point:", reference_points[-1])
    tip_point_coo = get_center_point(confident_tip['bbox'])
    tip_point_angle_calibrated = get_angle(tip_point_coo[0] - base_point[0], tip_point_coo[1] - base_point[1]) + calibration_value
    print("Tip Point:", tip_point_angle_calibrated)
    left = max((pt for pt in reference_points if pt['angle'] <= tip_point_angle_calibrated), key=lambda x: x['angle'], default=None)
    right = min((pt for pt in reference_points if pt['angle'] >= tip_point_angle_calibrated), key=lambda x: x['angle'], default=None)
    if left and right and left['angle'] != right['angle']:
        aL, vL = left['angle'], float(left['value'])
        aR, vR = right['angle'], float(right['value'])
        aT = tip_point_angle_calibrated
        value_tip = vL + (vR - vL) * (aT - aL) / (aR - aL)
    else:
        value_tip = None  # 无法插值

    return value_tip

def get_calibration(base_bbox: list, maximum_bbox: list, minimum_bbox: list) -> float:
    """
    计算角度校准值，假设 base_bbox 是基准点检测框，maximum_bbox 和 minimum_bbox 分别是最大值和最小值的检测框。
    返回角度校准值。
    """
    if not base_bbox or not maximum_bbox or not minimum_bbox:
        raise ValueError("Base, maximum, and minimum bounding boxes must be provided.")
    if len(base_bbox) != 4 or len(maximum_bbox) != 4 or len(minimum_bbox) != 4:
        raise ValueError("Bounding boxes must be in the format [x1, y1, x2, y2].")
    # 解包边界框坐标
    base_x1, base_y1, base_x2, base_y2 = base_bbox
    max_x1, max_y1, max_x2, max_y2 = maximum_bbox
    min_x1, min_y1, min_x2, min_y2 = minimum_bbox

    # 计算基准点的中心坐标
    base_cx, base_cy = get_center_point(base_bbox)

    # 计算最大值和最小值的中心坐标
    max_cx, max_cy = get_center_point(maximum_bbox)
    min_cx, min_cy = get_center_point(minimum_bbox)

    # 计算最大值和最小值坐标的中点坐标
    mid_cx = (max_cx + min_cx) / 2
    mid_cy = (max_cy + min_cy) / 2

    # 计算校准值（示例：使用最大值和最小值的距离）
    calibration_value = -get_angle(mid_cx - base_cx, mid_cy - base_cy)

    return calibration_value


def main_process(input_image_path: str, model_path: str):
    model = YOLO(model_path)
    results = model.predict(source=input_image_path, show=False, save=True, imgsz=640, conf=0.3)
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
                pil_image = Image.open(input_image_path).convert("RGB")
                cropped_img = pil_image.crop((x1, y1, x2, y2))
                ocr_result = numbers_ocr(cropped_img)
                pred["ocr_result"] = ocr_result
            preds.append(pred)

    calibration_value = get_calibration(
        base_bbox=next((pred['bbox'] for pred in preds if pred['name'] == 'base'), None),
        maximum_bbox=next((pred['bbox'] for pred in preds if pred['name'] == 'maximum'), None),
        minimum_bbox=next((pred['bbox'] for pred in preds if pred['name'] == 'minimum'), None)
    )
    print("Calibration Value:", calibration_value)
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
        result = interpolate_value(base_point['bbox'], confident_numbers, confidnet_tip, calibration_value)
        print("Interpolated Value:", result)


if __name__ == '__main__':
    input_image_path = "./test_imgs/2.jpg"
    model_path = "/Users/well/Documents/PyCharmProjects/auto-gauge-read/best.pt"
    main_process(input_image_path, model_path)