# dataset/fomo_dataset.py

import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob


class FomoCompatibleDataset(Dataset):
    """
    一个与FoMo-Net输入格式兼容的通用数据集类。
    它能够读取配置文件，并为每个图像样本返回对应的全局光谱keys。
    """

    def __init__(self, data_dir, dataset_name, config, transform=None, in_chans=None, modality_label=None):
        """
        初始化数据集。

        参数:
        data_dir (str): 图像文件所在的目录。
        dataset_name (str): 数据集的名称，必须在config的'dataset_modality_index'中存在。
        config (dict): 从JSON文件加载的完整配置字典。
        transform (callable, optional): 应用于图像的Torchvision transform。
        in_chans (int, optional): 图像的输入通道数。如果为None，会尝试从config推断。
        """
        super().__init__()

        # 1. 验证输入
        if dataset_name not in config['dataset_modality_index']:
            raise ValueError(f"Dataset name '{dataset_name}' not found in the config's 'dataset_modality_index'.")

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.config = config
        self.transform = transform

        self.modality_label = modality_label

        # 2. 获取图像文件列表
        self.image_files = sorted(glob.glob(os.path.join(data_dir, '*')))
        if not self.image_files:
            print(f"警告: 在目录 {data_dir} 中没有找到任何图像文件。")

        # 3. 解析配置文件，构建从本地通道索引到全局key的映射
        self.local_to_global_keys = self._build_key_mapping()

        # 4. 确定输入通道数
        self.in_chans = in_chans if in_chans is not None else len(self.local_to_global_keys)
        print(f"数据集 '{self.dataset_name}' 被初始化，包含 {len(self.image_files)} 张图像，"
              f"每个图像有 {self.in_chans} 个通道。")
        print(f"本地通道索引 -> 全局Key 的映射: {self.local_to_global_keys}")
        print(f"--- FINAL DEBUG for '{self.dataset_name}' ---")
        print(f"  Resulting self.local_to_global_keys: {self.local_to_global_keys}")
        print(f"  Keys that will be returned by __getitem__: {list(self.local_to_global_keys.values())}")
        print(f"-------------------------------------------")

    def _build_key_mapping(self):
        """
        根据配置文件构建从本地通道索引到全局key的映射。
        """
        dataset_map = self.config['dataset_modality_index'][self.dataset_name]

        # 反转全局字典，方便通过名称查找key
        band_name_to_key = {name: int(key) for key, name in self.config['modality_channels'].items()}

        # 创建本地索引到全局key的映射
        # 假设本地通道索引与dataset_map中的顺序一致
        # 我们按本地索引排序
        sorted_local_map = sorted(dataset_map.items(), key=lambda item: item[1])

        key_mapping = {}
        for band_name, local_index in sorted_local_map:
            if band_name not in band_name_to_key:
                raise ValueError(f"波段 '{band_name}' 在数据集 '{self.dataset_name}' 中定义，"
                                 f"但在全局 'modality_channels' 字典中找不到。")
            global_key = band_name_to_key[band_name]
            key_mapping[local_index] = global_key

        return key_mapping

    def __len__(self):
        return len(self.image_files)

    # def __getitem__(self, idx):
    #     # print(f"DEBUG: In __getitem__ for {self.dataset_name}, modality_label is {self.modality_label}")
    #     image_path = self.image_files[idx]
    #
    #     # 根据通道数决定如何打开图像
    #     if self.in_chans == 1:
    #         image = Image.open(image_path).convert('L')
    #     elif self.in_chans == 3:
    #         image = Image.open(image_path).convert('RGB')
    #     else:
    #         raise NotImplementedError(f"加载 {self.in_chans} 通道图像的逻辑尚未实现。")
    #
    #     if self.transform:
    #         image_tensor = self.transform(image)
    #
    #     # 获取这个数据集固定的keys列表
    #     # 我们直接使用解析好的映射的values
    #     keys_list = []
    #     for i in range(self.in_chans):
    #         keys_list.append(self.local_to_global_keys[i])
    #
    #     if self.modality_label is not None:
    #         return image_tensor, keys_list, self.modality_label
    #     else:
    #         # 保持向后兼容
    #         return image_tensor, keys_list
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')  # 总是转RGB

        if self.transform:
            image_tensor = self.transform(image)

        return image_tensor, self.modality_label

