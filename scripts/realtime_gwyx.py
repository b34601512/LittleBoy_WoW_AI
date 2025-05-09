import mss, cv2, numpy as np
from ultralytics import YOLO
import torch # 确保导入 torch

# --- 配置区 ---
# 【重要】修改为你实际的 best.pt 文件路径！
MODEL_PATH = r"D:/wow_ai/runs/detect/gwyx_detect_run_s_ep200_neg/weights/best.pt" 

# 屏幕截取区域 (根据你的游戏窗口调整)
# top: 距离屏幕顶部的像素
# left: 距离屏幕左侧的像素
# width: 截取宽度
# height: 截取高度
# 确保这个区域覆盖了你的游戏主要画面
MONITOR_AREA = {"top": 0, "left": 0, "width": 1920, "height": 1080} 

CONFIDENCE_THRESHOLD = 0.30  # 只显示置信度高于 30% 的检测结果 (可以调整)
# --- 配置区结束 ---

print("正在加载模型...")
# 明确指定使用 GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用的设备: {device}")
model = YOLO(MODEL_PATH)
model.to(device) # 将模型移动到 GPU
print("模型加载完毕.")

# 获取类别名称 (假设你的 yaml 文件里 names: [gwyx])
class_names = model.names
print(f"模型能识别的类别: {class_names}")

print("按 ESC 键退出实时检测...")

with mss.mss() as sct:
    while True:
        # 截取屏幕指定区域
        img_bgra = np.array(sct.grab(MONITOR_AREA))
        # 将 BGRA 转换为 BGR (OpenCV 通常使用 BGR)
        img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)

        # 使用模型进行推理 (直接在 GPU 上推理)
        # stream=True 可能更节省内存，但对于单帧处理差异不大
        results = model.predict(source=img_bgr, device=device, conf=CONFIDENCE_THRESHOLD, verbose=False) 

        # 绘制检测结果
        # results[0].plot() 会返回一个带有绘制框的图像副本
        annotated_frame = results[0].plot()

        # 显示结果窗口
        cv2.imshow("WoW Giant Owl Detector (Press ESC to exit)", annotated_frame)

        # 检测按键，如果按下 ESC (ASCII 码 27) 则退出循环
        if cv2.waitKey(1) == 27:
            break

# 关闭所有 OpenCV 窗口
cv2.destroyAllWindows()
print("实时检测已退出。")