# wow_rl/utils/detector.py
from ultralytics import YOLO
import torch
import numpy as np # 导入 numpy

class Detector:
    # 非常重要：将此路径更改为指向你自己的最佳模型！
    DEFAULT_WEIGHT_PATH = r'D:\wow_ai\runs\detect\gwyx_detect_run_s_ep200_neg\weights\best.pt'

    def __init__(self, weight_path=DEFAULT_WEIGHT_PATH, device='cuda:0'):
        try:
            self.model = YOLO(weight_path)
            self.device = device
            print(f"Detector 使用权重初始化: {weight_path} 在设备: {device}")
        except Exception as e:
            print(f"使用权重 {weight_path} 初始化 Detector 时出错: {e}")
            raise e # 如果模型加载失败，重新抛出异常以停止程序

    @torch.no_grad() # 对推理性能很重要 - 禁用梯度计算
    def infer(self, frame, conf=0.25):
        # verbose=False 禁用 YOLO 打印检测信息
        results = self.model.predict(frame, device=self.device,
                                     imgsz=640, conf=conf, verbose=False)
        # 检查是否有结果以及结果是否有 'boxes' 属性
        if results and results[0] and hasattr(results[0], 'boxes'):
            boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1,y1,x2,y2]
            scores = results[0].boxes.conf.cpu().numpy()
            return boxes, scores
        else:
            # 如果没有检测到任何东西，返回空数组
            return np.array([]), np.array([])