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

sys.path.append('/home/xyjiang/Matching/LightGlue/lightglue')
from lightglue import LightGLue
from superpoint import SuperPoint
import matplotlib.pyplot as plt
from sift import SIFT
from disk import DISK
from alike import ALIKED
from dog_hardnet import DoGHardNet

feature_extractors = {
    'superpoint': SuperPoint,
    'disk': DISK,
    'aliked': ALIKED,
    'sift': SIFT,
    'doghardnet': DogHardNet
}

default_conf = {
    'feature': 'superpoint',
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
        self.features = config.get('features', 'superpoint')  # 默认使用'superpoint'
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
        print('132111111111111111111', self.superpoint(data))
        pred = self.extractor(data)
        # 转化为superpoint的输出格式
        if self.features == 'DoGHardNet':
            pred = self.convert_output_to_standard_format(pred)
        elif self.features == 'disk':
            pred = self.convert_output_to_standard_format(pred)
        elif self.features == 'aliked':
            pred = self.convert_output_to_standard_format(pred)  # 输出形式应该是相同的，可以共用
        elif self.features == 'sift':
            pred = self.convert_output_to_standard_format(pred)
        return pred

    def get_matches_and_confidence(self, source_img, target_img, source_kp_dict, target_kp_dict,
                                   preprocess_image=True):
        # 提取 source 和 target 的关键点和描述符
        source_keypoints = source_kp_dict['keypoints'][0].cpu().numpy()
        source_descriptors = source_kp_dict['descriptors'][0].cpu().numpy()
        source_scores = source_kp_dict['scores'][0].cpu().numpy()

        target_keypoints = target_kp_dict['keypoints'][0].cpu().numpy()
        target_descriptors = target_kp_dict['descriptors'][0].cpu().numpy()
        target_scores = target_kp_dict['scores'][0].cpu().numpy()

        # 准备 lightGlue 输入
        data = {
            'image0': {
                'keypoints0': torch.from_numpy(source_keypoints).unsqueeze(0).cuda(),
                'descriptors0': torch.from_numpy(source_descriptors).unsqueeze(0).cuda(),
                'image0': source_img,
            },
            'image1': {
                'keypoints1': torch.from_numpy(target_keypoints).unsqueeze(0).cuda(),
                'descriptors1': torch.from_numpy(target_descriptors).unsqueeze(0).cuda(),
                'image1': target_img,
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





    def convert_output_to_standard_format(self, pred):
        keypoints = pred['keypoints']
        descriptors = pred['descriptors']
        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
        }

    def convert_disk_output_to_superpoint_format(self, pred):
        keypoints = pred['keypoints']
        scores = pred['keypoint_scores']
        descriptors = pred['descriptors']

        # 将 (B, N, 2) 的 keypoints 转换为列表，每个元素形状为 (N, 2)
        keypoints_list = [keypoints[i] for i in range(keypoints.shape[0])]

        # 将 (B, N) 的 scores 转换为列表，每个元素形状为 (N,)
        scores_list = [scores[i] for i in range(scores.shape[0])]

        # 将 (B, N, D) 的 descriptors 转换为列表，每个元素形状为 (D, N)
        descriptors_list = [descriptors[i].permute(1, 0) for i in range(descriptors.shape[0])]

        return {
            'keypoints': keypoints_list,
            'scores': scores_list,
            'descriptors': descriptors_list
        }

    def convert_sift_output_to_superpoint_format(self, sift_output):
        keypoints = sift_output['keypoints']
        if 'keypoint_scores' in sift_output:
            scores = sift_output['keypoint_scores']
        else:
            # 如果没有scores字段，设置默认值为1
            scores = torch.ones_like(keypoints[:, :, 0])
        descriptors = sift_output['descriptors']

        # 将 (B, N, 2) 的 keypoints 转换为列表，每个元素形状为 (N, 2)
        keypoints_list = [keypoints[i] for i in range(keypoints.shape[0])]

        # 将 (B, N) 的 scores 转换为列表，每个元素形状为 (N,)
        scores_list = [scores[i] for i in range(scores.shape[0])]

        # 将 (B, N, D) 的 descriptors 转换为列表，并将每个元素的形状从 (N, D) 转换为 (D, N)
        descriptors_list = [descriptors[i].permute(1, 0) for i in range(descriptors.shape[0])]

        return {
            'keypoints': keypoints_list,
            'scores': scores_list,
            'descriptors': descriptors_list
        }
