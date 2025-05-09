# wow_rl/buffers/replay_buffer.py
import numpy as np
import os
from pathlib import Path

class ReplayBuffer:
    def __init__(self, capacity, obs_shape, action_dim, save_dir='replay_buffer_data'):
        self.capacity = int(capacity)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True) # 创建保存目录

        # 使用 NumPy 内存映射文件 (memmap) 来高效存储大数据，避免内存溢出
        # 注意：obs 存储为 uint8 (0-255)，reward 存储为 float32
        print(f"Initializing Replay Buffer with capacity {self.capacity}")
        print(f"Observation shape: {obs_shape}, Action dimension: {action_dim}")
        print(f"Data will be saved in: {self.save_dir.resolve()}")

        try:
            self.obs = np.memmap(self.save_dir / 'obs.memmap', dtype=np.uint8, mode='w+', shape=(self.capacity,) + obs_shape)
            self.next_obs = np.memmap(self.save_dir / 'next_obs.memmap', dtype=np.uint8, mode='w+', shape=(self.capacity,) + obs_shape)
            self.actions = np.memmap(self.save_dir / 'actions.memmap', dtype=np.int64, mode='w+', shape=(self.capacity, action_dim)) # 动作通常是整数索引
            self.rewards = np.memmap(self.save_dir / 'rewards.memmap', dtype=np.float32, mode='w+', shape=(self.capacity, 1))
            self.dones = np.memmap(self.save_dir / 'dones.memmap', dtype=bool, mode='w+', shape=(self.capacity, 1))
            print("Memory-mapped files created successfully.")
        except Exception as e:
            print(f"Error creating memory-mapped files: {e}")
            print("Please check disk space and permissions.")
            raise e

        self._idx = 0 # 当前写入位置
        self._size = 0 # 当前存储的数据量
        self.meta_file = self.save_dir / 'buffer_meta.npz' # 用于保存 _idx 和 _size

    def add(self, obs, action, reward, next_obs, done):
        """添加一条经验到缓冲区"""
        try:
            # 确保动作也是 (1,) 数组 (更稳妥)
            action_arr = np.array([action], dtype=np.int64)

            # 写入数据到内存映射文件
            self.obs[self._idx] = obs.astype(np.uint8)
            self.actions[self._idx] = action_arr # 赋值 (1,) 数组
            # !! 修正 reward 赋值 !!
            self.rewards[self._idx] = np.array([reward], dtype=np.float32) # 创建 (1,) 数组
            self.next_obs[self._idx] = next_obs.astype(np.uint8)
            # !! 修正 done 赋值 !!
            self.dones[self._idx] = np.array([done], dtype=bool) # 创建 (1,) 数组

            # 更新索引和大小
            self._idx = (self._idx + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

        except Exception as e:
            print(f"Error adding data to replay buffer at index {self._idx}: {e}")
            print(f"obs type: {type(obs)}, shape: {getattr(obs, 'shape', 'N/A')}, dtype: {getattr(obs, 'dtype', 'N/A')}")
            print(f"action type: {type(action)}, value: {action}") # 打印原始 action
            print(f"reward type: {type(reward)}, value: {reward}") # 打印原始 reward
            print(f"next_obs type: {type(next_obs)}, shape: {getattr(next_obs, 'shape', 'N/A')}, dtype: {getattr(next_obs, 'dtype', 'N/A')}")
            print(f"done type: {type(done)}, value: {done}") # 打印原始 done
            # 如果需要，取消下面一行的注释以查看详细错误
            # traceback.print_exc()
            # 可以选择是否在这里重新抛出异常
            # raise e

    def save(self):
        """保存缓冲区元数据（大小和索引）"""
        try:
            print(f"Saving replay buffer metadata to {self.meta_file}...")
            # 确保 memmap 文件已刷新到磁盘
            self.obs.flush()
            self.next_obs.flush()
            self.actions.flush()
            self.rewards.flush()
            self.dones.flush()
            # 保存元数据
            np.savez(self.meta_file, idx=self._idx, size=self._size)
            print(f"Metadata saved: size={self._size}, idx={self._idx}")
        except Exception as e:
            print(f"Error saving replay buffer metadata: {e}")

    def load(self):
        """加载缓冲区元数据"""
        try:
            if self.meta_file.exists():
                meta = np.load(self.meta_file)
                self._idx = int(meta['idx'])
                self._size = int(meta['size'])
                print(f"Replay buffer metadata loaded: size={self._size}, idx={self._idx}")
            else:
                print(f"Metadata file {self.meta_file} not found. Starting with an empty buffer.")
                self._idx = 0
                self._size = 0
        except Exception as e:
            print(f"Error loading replay buffer metadata: {e}")
            self._idx = 0
            self._size = 0

    def __len__(self):
        return self._size

    # --- 用于 Dreamer 或其他库的数据集接口 ---
    def sample_batch(self, batch_size):
        """随机采样一个批次的数据"""
        if self._size < batch_size:
             print(f"Warning: Buffer size ({self._size}) is less than batch size ({batch_size}). Cannot sample.")
             return None # 或者抛出异常

        idxs = np.random.randint(0, self._size, size=batch_size)
        # 注意：需要从 memmap 读取数据
        batch = {
            'obs': self.obs[idxs].copy(),
            'action': self.actions[idxs].copy(),
            'reward': self.rewards[idxs].copy(),
            'next_obs': self.next_obs[idxs].copy(),
            'done': self.dones[idxs].copy()
        }
        return batch

    def as_tf_dataset(self, batch_size, sequence_length):
         """将缓冲区适配为 TensorFlow Dataset (需要 tensorflow)"""
         # 这个实现比较复杂，需要 tf.data.Dataset.from_generator
         # 暂时留空，DreamerV3 可能有自己的数据加载方式或需要特定格式
         print("Warning: as_tf_dataset is not fully implemented yet.")
         # 简单的示例生成器，需要适配 DreamerV3 的格式
         def _generator():
             while True:
                 idxs = np.random.randint(0, self._size - sequence_length + 1, size=batch_size)
                 # 需要根据序列长度来采样
                 # ... 实现序列采样逻辑 ...
                 # yield {'observation': seq_obs, 'action': seq_act, ...} # 调整键名以匹配 Dreamer
                 yield self.sample_batch(batch_size) # 临时返回批次

         # 需要指定输出签名 (output_signature)
         # output_signature = { 'observation': tf.TensorSpec(...), ... }
         # return tf.data.Dataset.from_generator(_generator, output_signature=...)
         return None

    def as_pytorch_dataloader(self, batch_size, sequence_length):
        """将缓冲区适配为 PyTorch DataLoader (需要 torch)"""
        # 这个实现相对直接，可以创建一个自定义的 Dataset 类
        # 暂时留空，DreamerV3 的 PyTorch 实现可能有自己的加载器
        print("Warning: as_pytorch_dataloader is not fully implemented yet.")
        from torch.utils.data import Dataset, DataLoader

        class BufferDataset(Dataset):
            def __init__(self, buffer, seq_len):
                self.buffer = buffer
                self.seq_len = seq_len

            def __len__(self):
                # 返回可以构成的完整序列的数量
                return max(0, len(self.buffer) - self.seq_len + 1)

            def __getitem__(self, index):
                # 返回一个序列
                end_idx = index + self.seq_len
                # 需要从 memmap 读取数据并转换为 torch tensor
                obs_seq = torch.tensor(self.buffer.obs[index:end_idx].copy(), dtype=torch.float32) / 255.0 # 归一化
                act_seq = torch.tensor(self.buffer.actions[index:end_idx].copy(), dtype=torch.long)
                rew_seq = torch.tensor(self.buffer.rewards[index:end_idx].copy(), dtype=torch.float32)
                done_seq = torch.tensor(self.buffer.dones[index:end_idx].copy(), dtype=torch.bool)
                # 可能还需要 next_obs，取决于 DreamerV3 实现
                # 需要调整字典键名以匹配 DreamerV3 的期望输入
                return {'observation': obs_seq, 'action': act_seq, 'reward': rew_seq, 'is_terminal': done_seq}

        dataset = BufferDataset(self, sequence_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 简化处理
        return dataloader