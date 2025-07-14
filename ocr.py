import easyocr
import cv2
import numpy as np
from PIL import Image

def numbers_ocr(image_input):
    """
    使用 EasyOCR 识别图片中的数字
    :param image_path: 图片路径
    :return: 识别结果列表
    """
    # 初始化 OCR 识别器
    reader = easyocr.Reader(['en'], gpu=True)

    # 判断输入类型并转换为 OpenCV 格式（BGR）
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    elif isinstance(image_input, Image.Image):
        img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("image_input must be a file path or PIL.Image.Image object")

    # 执行 OCR 识别
    results = reader.readtext(img)

    # 提取识别结果
    recognized_texts = []
    for bbox, text, conf in results:
        recognized_texts.append({"content": text, "conf": round(float(conf), 4)})

    return recognized_texts

if __name__ == '__main__':
    # 测试图片路径
    image_path = '/Users/well/Documents/PyCharmProjects/yolo-model-train/pressure gauge.v1i.yolov12/test/images/frames_0_4_83_jpg.rf.6056819351ddace338644bf491d53cdf.jpg'
    # 调用 OCR 函数
    recognized_texts = numbers_ocr(image_path)
    # 打印识别结果
    print(recognized_texts)