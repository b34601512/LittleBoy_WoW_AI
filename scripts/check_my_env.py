# D:\wow_ai\scripts\check_my_env.py
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import sys
import os
import traceback # 导入 traceback

# --- 动态路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) if os.path.basename(current_dir) == 'scripts' else current_dir
wow_rl_path = os.path.join(parent_dir, 'wow_rl')
if wow_rl_path not in sys.path:
    sys.path.append(parent_dir)
    print(f"Added {parent_dir} to sys.path")
# --- 路径设置结束 ---

try:
    from wow_rl.envs.wow_click_env import WowClickEnv
    print("Imported WowClickEnv successfully.")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)
except Exception as e: # 捕获其他可能的导入或初始化错误
    print(f"An unexpected error occurred during import or initialization:")
    traceback.print_exc()
    sys.exit(1)


print("Creating WowClickEnv instance for checking...")
# 创建环境实例，使用默认参数，不需要渲染
try:
    env_instance = WowClickEnv(render_mode=None)
    print("Environment instance created.")
except Exception as e:
    print(f"Error creating WowClickEnv instance:")
    traceback.print_exc()
    sys.exit(1)


# --- 运行环境检查 ---
print("\nRunning environment checker...")
try:
    # check_env 会自动调用 reset, step 等方法进行测试
    check_env(env_instance.unwrapped) # 使用 .unwrapped 获取原始环境实例
    print("\nEnvironment check passed successfully!")
except Exception as e:
    print("\nEnvironment check failed:")
    # 打印详细的错误信息
    traceback.print_exc()
finally:
    # 确保关闭环境
    try:
        env_instance.close()
    except Exception:
        pass # 忽略关闭时的错误

print("\nCheck finished.")