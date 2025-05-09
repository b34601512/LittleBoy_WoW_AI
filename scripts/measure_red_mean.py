import cv2
import numpy as np
import os # 导入 os 模块用于检查文件是否存在

# --- 配置参数 ---
# !! 非常重要：确保这里的 ROI 和你的 reward_sensor.py 中的一致 !!
ROI = (319, 35, 245, 64) # (x_left_top, y_left_top, width, height)

# !! 非常重要：确保这里的路径指向你保存的两张截图 !!
VISIBLE_IMAGE_PATH = r'D:\wow_ai\data\target_visible.png'
INVISIBLE_IMAGE_PATH = r'D:\wow_ai\data\no_target.png'
# --- 配置结束 ---

def calculate_roi_red_mean(image_path, roi):
    """加载图像，裁剪 ROI，并计算平均红色值"""
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：找不到图像文件 '{image_path}'")
        return None

    # 加载图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法加载图像 '{image_path}'")
        return None

    # 提取 ROI 坐标和尺寸
    x, y, w, h = roi

    # 确保 ROI 不超出图像边界
    img_h, img_w = img.shape[:2]
    roi_x_end = min(x + w, img_w)
    roi_y_end = min(y + h, img_h)
    roi_x_start = max(0, x)
    roi_y_start = max(0, y)

    # 检查裁剪区域是否有效
    if roi_x_end <= roi_x_start or roi_y_end <= roi_y_start:
        print(f"警告：对于图像 '{image_path}'，计算出的 ROI 为空或无效。")
        return 0 # 返回 0 或其他默认值

    # 裁剪 ROI
    crop = img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    # 检查裁剪区域是否为空
    if crop.size == 0:
        print(f"警告：对于图像 '{image_path}'，ROI 裁剪区域为空。")
        return 0 # 返回 0 或其他默认值

    # 计算红色通道 (BGR 索引为 2) 的平均值
    # 检查 crop 是否至少有一个维度大于0，避免 numpy 警告
    if crop.shape[0] > 0 and crop.shape[1] > 0:
        red_mean = np.mean(crop[:, :, 2])
        return red_mean
    else:
        print(f"警告：对于图像 '{image_path}'，裁剪区域维度无效。")
        return 0

# --- 主程序 ---
print(f"正在为 ROI {ROI} 计算平均红色值...")

# 计算“可见”状态的平均红色值
red_mean_visible = calculate_roi_red_mean(VISIBLE_IMAGE_PATH, ROI)
if red_mean_visible is not None:
    print(f"状态 '可见' (选中目标) 的平均红色值: {red_mean_visible:.2f}")

# 计算“不可见”状态的平均红色值
red_mean_invisible = calculate_roi_red_mean(INVISIBLE_IMAGE_PATH, ROI)
if red_mean_invisible is not None:
    print(f"状态 '不可见' (未选中目标) 的平均红色值: {red_mean_invisible:.2f}")

# 给出选择阈值的建议
if red_mean_visible is not None and red_mean_invisible is not None:
    suggested_threshold = (red_mean_visible + red_mean_invisible) / 2
    print(f"\n建议阈值 (Threshold) 可以选择两者之间的值，例如: {suggested_threshold:.2f}")
    print(f"请将选定的阈值填入 wow_rl/utils/reward_sensor.py 的 DEFAULT_RED_THRESHOLD")
else:
    print("\n无法计算建议阈值，请检查图像路径和 ROI 设置。")