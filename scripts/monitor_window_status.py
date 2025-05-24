# D:\wow_ai\scripts\monitor_window_status.py
"""
魔兽世界窗口状态监控脚本
用于确保AI训练期间游戏窗口保持活跃状态
"""

import time
import sys
import os
from datetime import datetime

# 导入窗口管理库
try:
    import pygetwindow as gw
    HAVE_WINDOW_UTILS = True
except ImportError:
    HAVE_WINDOW_UTILS = False
    print("❌ 错误: 未找到pygetwindow库，无法监控窗口状态")
    print("请安装: pip install pygetwindow")
    sys.exit(1)

class WindowMonitor:
    def __init__(self):
        self.wow_window = None
        self.last_active_state = None
        self.warning_count = 0
        self.start_time = time.time()
        
    def find_wow_window(self):
        """查找魔兽世界窗口"""
        try:
            # 按优先级搜索窗口标题
            search_patterns = ['World of Warcraft', 'Warcraft', '魔兽世界']
            
            for pattern in search_patterns:
                wow_windows = [w for w in gw.getAllTitles() if pattern in w]
                if wow_windows:
                    target_title = wow_windows[0]
                    wow_window_list = gw.getWindowsWithTitle(target_title)
                    if wow_window_list:
                        self.wow_window = wow_window_list[0]
                        print(f"✅ 找到WoW窗口: '{target_title}'")
                        return True
            
            print("❌ 未找到WoW窗口，请确保游戏正在运行")
            return False
            
        except Exception as e:
            print(f"❌ 查找窗口时出错: {e}")
            return False
    
    def check_window_status(self):
        """检查窗口当前状态"""
        if not self.wow_window:
            return False, "窗口未找到"
        
        try:
            # 刷新窗口信息
            self.wow_window = gw.getWindowsWithTitle(self.wow_window.title)[0]
            
            is_active = self.wow_window.isActive
            is_minimized = self.wow_window.isMinimized
            is_maximized = self.wow_window.isMaximized
            
            # 判断窗口状态
            if is_minimized:
                return False, "窗口已最小化"
            elif not is_active:
                return False, "窗口失去焦点"
            else:
                return True, "窗口状态正常"
                
        except Exception as e:
            return False, f"检查窗口状态出错: {e}"
    
    def log_status(self, is_ok, status_msg):
        """记录状态信息"""
        current_time = datetime.now().strftime("%H:%M:%S")
        elapsed = int(time.time() - self.start_time)
        elapsed_str = f"{elapsed//60:02d}:{elapsed%60:02d}"
        
        if is_ok:
            if self.last_active_state != True:
                print(f"✅ [{current_time}] (运行{elapsed_str}) 窗口状态恢复正常")
                if self.warning_count > 0:
                    print(f"📊 本次会话警告次数: {self.warning_count}")
        else:
            self.warning_count += 1
            print(f"⚠️  [{current_time}] (运行{elapsed_str}) 警告#{self.warning_count}: {status_msg}")
            print(f"   🔧 建议: 立即切换回游戏窗口或按F9暂停训练")
        
        self.last_active_state = is_ok
    
    def run_monitor(self, check_interval=2):
        """运行监控循环"""
        print("=" * 60)
        print(" 🔍 魔兽世界窗口状态监控器 ")
        print("=" * 60)
        print("此脚本将持续监控WoW窗口状态，确保AI训练期间窗口保持活跃。")
        print("如果窗口失去焦点或被最小化，会立即发出警告。")
        print("-" * 60)
        
        # 查找WoW窗口
        if not self.find_wow_window():
            print("❌ 无法找到WoW窗口，监控退出")
            return
        
        print(f"🔍 开始监控 (每{check_interval}秒检查一次)")
        print("💡 提示: 按 Ctrl+C 停止监控")
        print("=" * 60)
        
        try:
            while True:
                is_ok, status_msg = self.check_window_status()
                self.log_status(is_ok, status_msg)
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print(f"\n📊 监控统计:")
            print(f"   - 总运行时间: {int(time.time() - self.start_time)}秒")
            print(f"   - 警告次数: {self.warning_count}")
            print("👋 监控已停止")
        except Exception as e:
            print(f"❌ 监控过程中出错: {e}")

def main():
    """主函数"""
    monitor = WindowMonitor()
    monitor.run_monitor()

if __name__ == "__main__":
    main() 