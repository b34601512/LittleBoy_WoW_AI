# wow_rl/utils/interactor.py
import pydirectinput
import time
import random

class Interactor:
    # 假设屏幕是 1920x1080，并将其划分为 32x18 的网格
    def __init__(self, origin=(0,0), grid=(32,18), size=(1920,1080)):
        self.ox, self.oy = origin # 游戏区域的左上角坐标（通常是 0,0）
        self.gw, self.gh = grid   # 网格的维度 (宽, 高)
        self.sw, self.sh = size   # 游戏区域的像素尺寸 (宽, 高)
        # 计算单个网格单元的尺寸
        self.cell_w = self.sw // self.gw
        self.cell_h = self.sh // self.gh
        # pyautogui.PAUSE = 0 # 禁用 pyautogui 的暂停（pydirectinput 默认没有）

    def click_action(self, a:int):
        """ 在与动作 'a' 对应的网格单元执行点击 """
        if not (0 <= a < self.gw * self.gh):
            print(f"警告: 动作 {a} 超出范围 (0-{self.gw * self.gh - 1}). 跳过点击。")
            return

        # 根据动作 'a' 计算网格坐标 (gx, gy)
        gx = a % self.gw
        gy = a // self.gw

        # 计算网格单元中心的像素坐标
        cell_center_x = self.ox + gx * self.cell_w + self.cell_w // 2
        cell_center_y = self.oy + gy * self.cell_h + self.cell_h // 2

        # 添加小的随机偏移，避免总是点击完全相同的像素
        # 将随机性限制在例如单元格尺寸的 +/- 1/4
        rand_x_offset = random.randint(-self.cell_w // 4, self.cell_w // 4)
        rand_y_offset = random.randint(-self.cell_h // 4, self.cell_h // 4)

        click_x = cell_center_x + rand_x_offset
        click_y = cell_center_y + rand_y_offset

        # 确保点击坐标在屏幕边界内
        click_x = max(self.ox, min(click_x, self.ox + self.sw - 1))
        click_y = max(self.oy, min(click_y, self.oy + self.sh - 1))

        # print(f"动作: {a} -> 网格({gx},{gy}) -> 点击坐标 ({click_x},{click_y})") # 用于调试

        # 使用 pydirectinput 移动和点击
        try:
            pydirectinput.moveTo(click_x, click_y)
            pydirectinput.click()
            time.sleep(0.05)  # 点击后短暂暂停
        except Exception as e:
            print(f"pydirectinput 交互期间出错: {e}")