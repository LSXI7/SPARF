from copy import deepcopy
from pathlib import Path
from typing import List, Tuple
import tarfile
import torch
from torch import nn
import sys
import cv2
import os
from PIL import Image
import numpy as np
import h5py
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict, Any, Optional

from match_methods.LightGlue.lightglue.lightglue import LightGlue
from match_methods.LightGlue.lightglue.superpoint import SuperPoint
from match_methods.LightGlue.lightglue.sift import SIFT
from match_methods.LightGlue.lightglue.disk import DISK
from match_methods.LightGlue.lightglue.aliked import ALIKED
from match_methods.LightGlue.lightglue.dog_hardnet import DoGHardNet

feature_extractors = {
    'superpoint': SuperPoint,
    'disk': DISK,
    'aliked': ALIKED,
    'sift': SIFT,
    'doghardnet': DoGHardNet,
}

default_conf = {
    'features': 'superpoint',
    'lightglue': {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": 0.95,  # early stopping, disable with -1
        "width_confidence": 0.99,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    }
}


class SPLGInference(nn.Module):
    def __init__(self, config=default_conf):
        super().__init__()
        self.features = config.get('features', {})  # 默认使用'superpoint'
        print("使用特征提取器为：", self.features)
        lightglue_config = config.get('lightglue', {})
        self.lightglue = LightGlue(self.features, **lightglue_config).eval().cuda()
        if self.features in feature_extractors:
            self.extractor = feature_extractors[self.features]().eval().cuda()
        else:
            raise ValueError(f"Unsupported features: {features}")

    def pre_process_img(self, images):
        device = images.device  # 获取输入图像所在的设备
        gray_images = []
        for image in images:
            # 将单个图像转移到CPU
            image = image.cpu().numpy()
            # 转换形状为 (H, W, C)
            image = np.transpose(image, (1, 2, 0))
            # 将图像转换为灰度图像
            img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # 将灰度图像转换为numpy数组并调整形状为 (1, H, W)
            gray_array = img.astype(np.float32) / 255.0
            # 将灰度图像转换为 PyTorch 张量并调整形状为 (1, H, W)
            gray_tensor = torch.tensor(gray_array, dtype=torch.float32, device=device).unsqueeze(0)
            gray_images.append(gray_tensor)

        # 将所有灰度图像堆叠成一个形状为 (B, 1, H, W) 的张量
        gray_images = torch.stack(gray_images)
        return gray_images

    def get_keypoints(self, images):
        data = {'image': images}
        print(f"成功处理图片，方式为: {type(self.extractor).__name__}")
        pred = self.extractor(data)
        # print('132111111111111111111', pred)
        return pred

    def get_matches_and_confidence(self, source_img, target_img, source_kp_dict, target_kp_dict,
                                   preprocess_image=True):
        # 提取 source 和 target 的关键点和描述符
        source_keypoints = source_kp_dict['keypoints'][0].cpu().numpy()
        print('source_keypoints:', source_keypoints.shape)
        source_descriptors = source_kp_dict['descriptors'][0].cpu().numpy()
        # source_scores = source_kp_dict['scores'][0].cpu().numpy()

        target_keypoints = target_kp_dict['keypoints'][0].cpu().numpy()
        print('target_keypoints:', target_keypoints.shape)
        target_descriptors = target_kp_dict['descriptors'][0].cpu().numpy()
        print('self.lightglue.conf.add_scale_ori:', self.lightglue.conf.add_scale_ori)
        if self.lightglue.conf.add_scale_ori:
            source_scales = source_kp_dict['scales'][0].cpu().numpy()
            source_oris = source_kp_dict['oris'][0].cpu().numpy()
            target_scales = target_kp_dict['scales'][0].cpu().numpy()
            target_oris = target_kp_dict['oris'][0].cpu().numpy()

        # target_scores = target_kp_dict['scores'][0].cpu().numpy()
        if self.features == "superpoint":
            print('source_keypoints:', source_keypoints.shape)
            print('source_descriptors:', source_descriptors.shape)
            source_descriptors = source_descriptors.transpose(1, 0)
            target_descriptors = target_descriptors.transpose(1, 0)

        # 准备 lightGlue 输入
        data = {
            'image0': {
                'keypoints': torch.from_numpy(source_keypoints).unsqueeze(0).cuda(),
                'descriptors': torch.from_numpy(source_descriptors).unsqueeze(0).cuda(),
                'image': source_img,
            },
            'image1': {
                'keypoints': torch.from_numpy(target_keypoints).unsqueeze(0).cuda(),
                'descriptors': torch.from_numpy(target_descriptors).unsqueeze(0).cuda(),
                'image': target_img,
            }
        }
        if self.lightglue.conf.add_scale_ori:
            data = {
                'image0': {
                    'keypoints': torch.from_numpy(source_keypoints).unsqueeze(0).cuda(),
                    'descriptors': torch.from_numpy(source_descriptors).unsqueeze(0).cuda(),
                    'image': source_img,
                    'scales': torch.from_numpy(source_scales).unsqueeze(0).cuda(),
                    'oris': torch.from_numpy(source_oris).unsqueeze(0).cuda(),
                },
                'image1': {
                    'keypoints': torch.from_numpy(target_keypoints).unsqueeze(0).cuda(),
                    'descriptors': torch.from_numpy(target_descriptors).unsqueeze(0).cuda(),
                    'image': target_img,
                    'scales': torch.from_numpy(target_scales).unsqueeze(0).cuda(),
                    'oris': torch.from_numpy(target_oris).unsqueeze(0).cuda(),
                }
            }

        # 运行 lightGlue 模型
        pred = self.lightglue(data)

        # 从 lightGlue 输出中提取匹配信息
        matches0 = pred['matches0'][0].cpu().numpy()  # 从 source 到 target 的匹配索引
        matching_scores0 = pred['matching_scores0'][0].cpu().numpy()  # 匹配的置信度

        valid = matches0 > -1
        mkpts0 = source_keypoints[valid]
        mkpts1 = target_keypoints[matches0[valid]]
        mconf = matching_scores0[valid]

        # 返回结果字典
        return {
            'kp_source': mkpts0,
            'kp_target': mkpts1,
            'confidence_value': mconf,
        }

    def visualize_keypoints(self, images, keypoints_data, output_dir='kp_output'):
        print('成功')
        print('images:', images.shape)

        # 自动创建输出目录
        next_output_dir = self.create_output_directory(output_dir)

        images = images.cpu().numpy()
        for i, (image, keypoints) in enumerate(zip(images, keypoints_data['keypoints'])):
            fig, ax = plt.subplots()

            # 转换图像为灰度图像
            if image.shape[0] == 3:
                image = 0.2989 * image[0, :, :] + 0.5870 * image[1, :, :] + 0.1140 * image[2, :, :]

            ax.imshow(image, cmap='gray')
            if keypoints.numel() > 0:  # 确保关键点存在
                ax.scatter(keypoints[:, 1].cpu(), keypoints[:, 0].cpu(), c='r',
                           s=5)  # 注意这里keypoints[:, 1]是x坐标，keypoints[:, 0]是y坐标
            ax.set_title(f"Image {i + 1}")
            ax.axis('off')

            output_path = os.path.join(next_output_dir, f"image_{i + 1}.png")
            plt.savefig(output_path)
            plt.close(fig)  # 关闭图形，以便在循环中不会累积

    def plot_matches(self, source_img, target_img, kp_source, kp_target, output_path):
        source_img = source_img.cpu().numpy().transpose(1, 2, 0)
        target_img = target_img.cpu().numpy().transpose(1, 2, 0)

        # 如果图像是3通道，转换为灰度图像
        if source_img.shape[2] == 3:
            source_img = 0.2989 * source_img[:, :, 0] + 0.5870 * source_img[:, :, 1] + 0.1140 * source_img[:, :, 2]
        if target_img.shape[2] == 3:
            target_img = 0.2989 * target_img[:, :, 0] + 0.5870 * target_img[:, :, 1] + 0.1140 * target_img[:, :, 2]

        kp_source = kp_source.cpu().numpy()
        kp_target = kp_target.cpu().numpy()

        # 拼接两张图像
        combined_img = np.concatenate((source_img, target_img), axis=1)

        # 调整关键点位置
        kp_target_adjusted = kp_target.copy()
        kp_target_adjusted[:, 0] += source_img.shape[1]

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(combined_img, cmap='gray')
        ax.scatter(kp_source[:, 0], kp_source[:, 1], c='r', s=5)
        ax.scatter(kp_target_adjusted[:, 0], kp_target_adjusted[:, 1], c='r', s=5)

        for i in range(len(kp_source)):
            ax.plot([kp_source[i, 0], kp_target_adjusted[i, 0]], [kp_source[i, 1], kp_target_adjusted[i, 1]], "r-",
                    linewidth=0.5)

        ax.axis('off')
        plt.savefig(output_path)
        plt.close(fig)

    def create_output_directory(self, base_dir='out'):
        # 自动创建输出目录
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # 查找下一个可用的数字命名文件夹
        subdirs = [int(d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
        next_dir = str(max(subdirs) + 1) if subdirs else '1'
        next_output_dir = os.path.join(base_dir, next_dir)
        os.makedirs(next_output_dir)

        return next_output_dir

    def extract_keypoints_and_save_h5(self, keypoint_extractor_module, images_dir, image_names, export_dir, name,
                                      overwrite=False):
        feature_path = Path(export_dir, name + '_keypoints.h5')
        feature_path.parent.mkdir(exist_ok=True, parents=True)
        if os.path.exists(str(feature_path)) and not overwrite:
            return feature_path

        feature_file = h5py.File(str(feature_path), 'w')

        print('Compute keypoints over a grid')
        pbar = tqdm(enumerate(image_names), total=len(image_names))
        for i, image_name in pbar:
            image = imageio.imread(os.path.join(images_dir, image_name))
            image = pre_process_img(image)
            kp = keypoint_extractor_module.get_keypoints(torch.tensor(image).unsqueeze(0).cuda())

            # Assuming kp is a dictionary with multiple keys
            grp = feature_file.create_group(image_name)
            for key, value in kp.items():
                value_np = torch.stack(value).cpu().numpy() if isinstance(value, list) else value.cpu().numpy()
                grp.create_dataset(key, data=value_np)
        feature_file.close()
        return feature_path

    def save_matches_to_h5(self, matches_dict, pair_names, export_dir, match_name, overwrite=False):
        match_path = os.path.join(export_dir, match_name + '.h5')
        if os.path.exists(match_path) and not overwrite:
            return Path(match_path)

        with h5py.File(match_path, 'w') as match_file:
            pbar = tqdm(enumerate(pair_names), total=len(pair_names))
            for i, pair in pbar:
                name0, name1 = pair
                name_of_pair = names_to_pair(name0, name1)
                grp = match_file.create_group(name_of_pair)
                matches = matches_dict[name_of_pair].numpy().astype(np.int32)  # Convert tensor to numpy array
                grp.create_dataset('matches', data=matches)

        print('Finished exporting matches.')
        return Path(match_path)

    def retrieve_matches_at_keypoints_locations_from_pair_list(self,
                                                               pair_names: List[str], images_dir: str,
                                                               name_to_pair_function: Callable[[Any], Any],
                                                               path_to_h5_keypoints: str = None,
                                                               key_for_keypoints: str = None,
                                                               matches_dict: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Retrieves matches between each image pair specificied in a list of image pairs, with prior keypoints extracted
        densely in a grid for each image.
        Each match has shape 1x2, it contains the index of the corresponding keypoints in source and target
        images respectively. It can also contain the confidence of the match, in that case the match is 1x3.

        Args:
            args: Arguments dictionary
            cfg: Config, check default_cfg
            pair_names: List of pair names
            images_dir: Directory containing images
            kp_dict: Dictionary containing keypoints for each image
            path_to_h5_keypoints: Path to h5 file containing keypoints for each image
            key_for_keypoints: Additional keys to access keypoint in kp_dict, when applicable
            matches_dict: Dictionary containing matches

        Returns:
            matches_dict: Dictionary containing matches, where there is a key for each image pair,
                          defined by the name_to_pair_function.
                          For each pair, Nx2 for N matches, contains the index of the corresponding keypoints
                          in source and target image respectively.
                          If a confidence value is available, Nx3, where the third value is the confidence of the match.
        """
        if path_to_h5_keypoints is not None:
            assert os.path.exists(path_to_h5_keypoints)
            kp_dict = h5py.File(path_to_h5_keypoints, 'r')

        # 打印出HDF5文件中的所有键，以检查目标键是否存在
        available_keys = list(kp_dict.keys())
        print(f"Available keys in HDF5 file: {available_keys}")

        pbar = tqdm(enumerate(pair_names), total=len(pair_names))
        for i, pair in pbar:
            img_fname0, img_fname1 = pair
            src_fn = os.path.join(images_dir, img_fname0)
            tgt_fn = os.path.join(images_dir, img_fname1)
            name_of_pair = names_to_pair(img_fname0, img_fname1)
            name_of_pair_for_flow = name_of_pair.replace('/', '-').replace(' ', '--')

            if name_of_pair in matches_dict:
                continue

            image0_original = imageio.imread(src_fn)[:, :, :3]
            image1_original = imageio.imread(tgt_fn)[:, :, :3]

            source_kp_dict = h5_to_tensor_dict(kp_dict[img_fname0])
            target_kp_dict = h5_to_tensor_dict(kp_dict[img_fname1])
            # print('\n\n\n\nsource_kp_dict:', source_kp_dict)
            # print('\n\n\n\ntarget_kp_dict:', target_kp_dict)
            matches = self.get_matches_and_confidence(
                source_img=torch.tensor(image0_original).permute(2, 0, 1).unsqueeze(0).cuda(),
                target_img=torch.tensor(image1_original).permute(2, 0, 1).unsqueeze(0).cuda(),
                source_kp_dict=source_kp_dict,
                target_kp_dict=target_kp_dict
            )
            # print('\n\n\n\nmatches:', matches)

            matches_dict[name_of_pair] = convert_matches_to_index_dict(matches, source_kp_dict['keypoints'],
                                                                       target_kp_dict['keypoints'],
                                                                       confidence_threshold=0.5)
            print('\n\n\n\nmatches_dict:', matches_dict)
        return matches_dict


def names_to_pair(name0: str, name1: str):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))


def pre_process_img(image):
    # 将单个图像转移到CPU
    # 将图像转换为灰度图像
    print(image.dtype)
    print(image.shape)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 将灰度图像转换为numpy数组并调整形状为 (1, H, W)
    gray_array = img.astype(np.float32) / 255.0
    # 将灰度图像转换为 PyTorch 张量并调整形状为 (1, H, W)
    gray_tensor = torch.tensor(gray_array, dtype=torch.float32).unsqueeze(0)
    print(gray_tensor.shape)
    return gray_tensor


def h5_to_tensor_dict(h5_group):
    result = {}
    for key in h5_group.keys():
        data = h5_group[key][...]
        tensor_data = torch.tensor(data)
        result[key] = tensor_data
        print(f"Key: {key}, Shape: {tensor_data.shape}")
    return result


def find_closest_index(coord, keypoints):
    """
    在 keypoints 中找到最接近 coord 的索引
    """
    distances = torch.norm(keypoints - torch.tensor(coord), dim=1)
    return torch.argmin(distances).item()


def convert_matches_to_index_dict(matches, source_kp_dict, target_kp_dict, confidence_threshold=0.5):
    # 提取 matches 中的 kp_source、kp_target 和 confidence_value
    kp_source_coords = matches['kp_source']
    print('kp_source_coords:', kp_source_coords.shape)
    kp_target_coords = matches['kp_target']
    print('kp_target_coords:', kp_target_coords.shape)
    confidence_values = matches['confidence_value']
    print('confidence_values:', confidence_values.shape)

    kp_source_coords = torch.tensor(kp_source_coords, dtype=torch.float32)
    print('kp_source_coords:', kp_source_coords.shape)
    kp_target_coords = torch.tensor(kp_target_coords, dtype=torch.float32)
    print('kp_target_coords:', kp_target_coords.shape)
    # 只保留置信度大于阈值的匹配点
    valid_indices = np.where(confidence_values > confidence_threshold)[0]
    kp_source_coords = kp_source_coords[valid_indices]
    kp_target_coords = kp_target_coords[valid_indices]

    # 获取 keypoints 并移除第一个维度
    # print('\n\n\n\nsource_kp_dict:', source_kp_dict)
    source_keypoints = torch.squeeze(source_kp_dict, dim=0)
    target_keypoints = torch.squeeze(target_kp_dict, dim=0)
    print('\n\n\n\nsource_keypoints:', source_keypoints.shape)
    print('\n\n\n\ntarget_keypoints:', target_keypoints.shape)

    source_indices = torch.zeros(kp_source_coords.shape[0], dtype=torch.long)
    for i, kp in enumerate(kp_source_coords):
        index = torch.all(source_keypoints == kp, dim=1).nonzero(as_tuple=True)[0]
        source_indices[i] = index

    # 找到kp_target_coords在target_keypoints中的索引
    target_indices = torch.zeros(kp_target_coords.shape[0], dtype=torch.long)
    for i, kp in enumerate(kp_target_coords):
        index = torch.all(target_keypoints == kp, dim=1).nonzero(as_tuple=True)[0]
        target_indices[i] = index

    # 组合索引
    combined_indices = torch.stack((source_indices, target_indices), dim=1)

    print(combined_indices)
    return combined_indices
