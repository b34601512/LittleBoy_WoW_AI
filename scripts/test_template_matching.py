import cv2
import numpy as np
import mss # 直接使用 mss 来截屏，更简单
import time
import os

# --- 配置参数 ---
# !! 确认这是正确的 ROI !!
ROI = (250, 35, 205, 64) # (x_left_top, y_left_top, width, height)

# !! 确认这是正确的模板路径 !!
TEMPLATE_PATH = r'D:\wow_ai\data\target_frame_template.png'

# 截取屏幕的区域 (可以只截取包含 ROI 的部分以提高效率，或者截全屏)
# 这里我们截取屏幕左上角 800x600 的区域，应该包含了你的 ROI
# 如果你的 ROI 在屏幕其他位置，需要调整这里的 region
# 或者直接截全屏: SCREEN_REGION = {'left': 0, 'top': 0, 'width': 1920, 'height': 1080}
SCREEN_REGION = {'left': 0, 'top': 0, 'width': 800, 'height': 600}

# 模板匹配方法
MATCH_METHOD = cv2.TM_CCOEFF_NORMED
# --- 配置结束 ---

print("--- 模板匹配独立测试脚本 ---")
print(f"ROI: {ROI}")
print(f"模板路径: {TEMPLATE_PATH}")

# 1. 加载模板图像
if not os.path.exists(TEMPLATE_PATH):
    print(f"错误：找不到模板文件 '{TEMPLATE_PATH}'")
    exit()
template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_COLOR) # 以彩色加载
if template is None:
    print(f"错误：无法加载模板 '{TEMPLATE_PATH}'")
    exit()
# 获取模板的宽度和高度，用于后面绘制矩形
th, tw = template.shape[:2]
print(f"模板加载成功，尺寸 (高x宽): {th}x{tw}")

# (可选) 如果想测试灰度匹配，取消下面两行注释
# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# print("模板已转换为灰度图")

# 2. 初始化屏幕截图工具
sct = mss.mss()
print(f"屏幕截图区域设置为: {SCREEN_REGION}")

# 3. 主循环：截屏、裁剪、匹配、显示
print("\n开始实时匹配... 按 'q' 键退出")
window_name = "Template Matching Test - ROI View"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # 创建可调整窗口

while True:
    # 记录开始时间，用于计算 FPS (可选)
    start_time = time.time()

    # a. 截取屏幕指定区域
    frame_full = np.array(sct.grab(SCREEN_REGION))
    # 转换为 BGR 格式
    frame_bgr = cv2.cvtColor(frame_full, cv2.COLOR_BGRA2BGR)

    # b. 从截屏中裁剪出 ROI 区域
    x, y, w, h = ROI
    # !! 注意：这里的 x, y 是相对于 SCREEN_REGION 左上角的 !!
    # 如果 SCREEN_REGION 是 (0,0,...)，那么 x, y 不变
    # 如果 SCREEN_REGION 不是从 (0,0) 开始，需要调整 x, y:
    # roi_x_start = max(0, x - SCREEN_REGION['left'])
    # roi_y_start = max(0, y - SCREEN_REGION['top'])
    roi_x_start = max(0, x) # 假设 SCREEN_REGION 从 0,0 开始
    roi_y_start = max(0, y)
    roi_x_end = min(roi_x_start + w, frame_bgr.shape[1])
    roi_y_end = min(roi_y_start + h, frame_bgr.shape[0])

    if roi_x_end <= roi_x_start or roi_y_end <= roi_y_start:
        print("警告：ROI 区域无效或超出截图范围，跳过此帧")
        time.sleep(0.1) # 避免空跑 CPU 占用过高
        continue

    roi_crop = frame_bgr[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    # 检查 ROI 是否有效，以及模板是否能放进去
    if roi_crop.size == 0 or template.shape[0] > roi_crop.shape[0] or template.shape[1] > roi_crop.shape[1]:
        print(f"警告：ROI 裁剪区域为空或模板 ({th}x{tw}) 大于 ROI ({roi_crop.shape[0]}x{roi_crop.shape[1]})，跳过此帧")
        # 在窗口显示提示信息
        display_roi = np.zeros((100, 300, 3), dtype=np.uint8) # 创建黑色背景
        cv2.putText(display_roi, "ROI Invalid or Too Small", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(window_name, display_roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue # 跳到下一个循环

    # (可选) 如果测试灰度匹配，取消下面一行注释
    # roi_crop_display = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY) # 转换用于匹配
    roi_crop_display = roi_crop # 保持彩色用于显示

    # c. 执行模板匹配
    result = cv2.matchTemplate(roi_crop_display, template, MATCH_METHOD)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 根据不同的匹配方法，最佳匹配点可能不同
    if MATCH_METHOD in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        match_loc = min_loc
        match_score = 1.0 - min_val # 将最小值转换为相似度 (值越小越好)
    else:
        match_loc = max_loc
        match_score = max_val # 值越大越好

    # d. 准备显示结果
    # 复制一份 ROI 图像用于绘制，避免修改原始数据
    display_roi = roi_crop.copy()

    # 在左上角显示匹配分数
    score_text = f"Score: {match_score:.4f}"
    cv2.putText(display_roi, score_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 在最佳匹配位置绘制一个矩形框
    # match_loc 是匹配到的模板左上角在 ROI 区域内的坐标
    bottom_right = (match_loc[0] + tw, match_loc[1] + th)
    cv2.rectangle(display_roi, match_loc, bottom_right, (0, 0, 255), 2) # 红色框

    # e. 显示带有分数的 ROI 图像
    cv2.imshow(window_name, display_roi)

    # f. 检测退出键 ('q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # (可选) 打印 FPS
    # end_time = time.time()
    # fps = 1 / (end_time - start_time)
    # print(f"FPS: {fps:.2f}")

# 清理
cv2.destroyAllWindows()
print("测试结束。")