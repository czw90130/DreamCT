import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import open3d as o3d
import SimpleITK as sitk
from tqdm import tqdm
import random


class SentenceBuilder:
    def age_description(self, age, digit=False):
        if digit:
            return str(age)
        if age < 25:
            return 'a young'
        elif 25 <= age < 35:
            return 'a young-adult'
        elif 35 <= age < 45:
            return 'an adult'
        elif 45 <= age < 55:
            return 'a middle-aged'
        elif 55 <= age < 65:
            return 'an elderly'
        elif 65 <= age < 75:
            return 'a senior'
        elif 75 <= age < 85:
            return 'an advanced-senior'
        elif age >= 85:
            return 'a long-lived-senior'
        else:
            return 'an unknown-age'

    def gender_description(self, gender):
        male_descriptions = ['male', 'man', 'gentleman']
        female_descriptions = ['female', 'woman', 'lady']

        if gender == 'm':
            # 随机选择一个男性描述词
            return random.choice(male_descriptions)
        elif gender == 'f':
            # 随机选择一个女性描述词
            return random.choice(female_descriptions)
        else:
            return 'unspecified-gender'

    def plane_full_name(self, plane, short=False):
        if short:
            return plane
        mapping = {'sag': 'sagittal', 'hor': 'horizontal', 'cor': 'coronal'}
        
        return mapping.get(plane, 'unknown')

    def spine_full_name(self, spine, short=False):
        if short:
            return spine
        spine_mapping = {
            'C': 'Cervical',
            'T': 'Thoracic',
            'L': 'Lumbar',
            'S': 'Sacrum',
        }
        try:
            tag = spine[0]
            name = spine_mapping[tag]
            if spine[1] == '0':
                return name
            else:
                return name + '-' + spine[1:]
        except:
            return spine

    def list_to_natural_language(self, lst):
        if not lst:
            return 'none'
        elif len(lst) == 1:
            return lst[0]
        else:
            return ', '.join(lst[:-1]) + ' and ' + lst[-1]
        
    def order_description(self, positive_order):
        if positive_order:
            return 'in a forward sequence'
        else: 
            return 'in a reverse sequence'

    def dict_to_sentence(self, d, randomize=True):
        if randomize:
            age_digit = random.choice([True, False])
            short_spine = random.choice([True, False])
            short_plane = random.choice([True, False])
        else:
            age_digit = False
            short_spine = False
            short_plane = False
        
        parts = {
            'age': self.age_description(d['age'],digit=age_digit) if 'age' in d else "",
            'gender': self.gender_description(d['gender']) if 'gender' in d else "",
            'plane': self.plane_full_name(d['plane'], short=short_plane) if 'plane' in d else "",
            'spine_list': self.list_to_natural_language([self.spine_full_name(spine, short=short_spine) for spine in d['spines'].split('|')]) if 'spines' in d else "",
            'order': self.order_description(d['positive_order']),
        }
        
        if not randomize:
            output = ''
            for k, v in parts.items():
                output += f"{k}:{v}, "
            return output[:-2]

        templates = [
            f"{parts['age']} {parts['gender']} patient has observations {parts['order']} in the {parts['plane']} plane, including {parts['spine_list']}.",
            f"In the {parts['plane']} plane {parts['order']}, the {parts['gender']} patient of {d.get('age', 'unknown-age')}-years shows the following spines: {parts['spine_list']}.",
            f"For {parts['age']} {parts['gender']}, the CT scan in the {parts['plane']} plane {parts['order']} reveals {parts['spine_list']}.",
            f"Observations for {parts['age']} {parts['gender']} {parts['order']}: {parts['spine_list']} in the {parts['plane']} plane.",
            f"{parts['age']} {parts['gender']} shows {parts['spine_list']} on {parts['plane']} plane imaging {parts['order']}.",
            f"{parts['plane'].capitalize()} plane analysis {parts['order']} reveals {parts['spine_list']} for this {parts['age']} {parts['gender']}.",
            f"CT findings: {parts['spine_list']} in the {parts['plane']} plane {parts['order']} for {parts['age']} {parts['gender']}.",
            f"Diagnosis for {parts['age']} {parts['gender']}: {parts['spine_list']}, as seen in the {parts['plane']} plane.",
            f"{parts['plane'].capitalize()} plane imaging {parts['order']} {('reveals ' + parts['spine_list']) if parts['spine_list'] else 'analysis'} for {(parts['age'] + ' ') if parts['age'] else ''}{parts['gender']} patient.",
            f"Patient details: {(', '.join(filter(None, [parts['age'], parts['gender'], parts['spine_list']])) + ',') if any([parts['age'], parts['gender'], parts['spine_list']]) else 'Not fully specified'}, observed in {parts['plane']} plane {parts['order']}.",
            f"{(parts['age'] + ' ') if parts['age'] else ''}{parts['gender']} with findings in the {parts['plane']} plane {parts['order']}: {parts['spine_list']}.",
            f"{('Findings for ' + parts['age'] + ' ') if parts['age'] else ''}{parts['gender']}: {parts['spine_list']} in the {parts['plane']} plane {parts['order']}.",
            f"CT scan {('in ' + parts['plane'] + ' plane ') if parts['plane'] else ''}{parts['order']} shows {parts['spine_list']} for {(parts['age'] + ' ') if parts['age'] else ''}{parts['gender']}.",
            f"Analysis{(': ' + parts['plane'] + ' plane, ') if parts['plane'] else ': '} {parts['order']} {parts['spine_list']} {'for ' + parts['age'] + ' ' + parts['gender'] if parts['age'] and parts['gender'] else ('for ' + (parts['age'] or parts['gender']))}.",
            f"{parts['age']} {parts['gender']}, {parts['plane']}: {parts['spine_list']}.",
            f"{parts['gender']} {parts['age']}, {parts['spine_list']} in {parts['plane']} {parts['order']}.",
            f"{parts['plane']} {parts['order']} - {parts['spine_list']}, {parts['age']} {parts['gender']}.",
            f"{parts['age']} {parts['gender']} {parts['plane']}: {parts['spine_list']}.",
            f"{parts['spine_list']} ({parts['plane']}, {parts['age']} {parts['gender']}) {parts['order']}.",
            f"{parts['age']} {parts['gender']}, {parts['plane']} plane {parts['order']}.",
            f"{parts['gender']} patient, {parts['spine_list']} {parts['order']}.",
            f"{parts['plane']} plane: {parts['spine_list']}.",
            f"{parts['age']} {parts['gender']}: {parts['spine_list']} {parts['order']}.",
            f"{parts['spine_list']}, {parts['plane']} plane {parts['order']}.",
            ""
        ]

        # 随机选择一个模板并返回
        return random.choice(templates).replace("  ", " ").replace(",,", ",").strip()


class NIfTIEncoder(SentenceBuilder):
    def __init__(self, hu_range=(-1000, 3000), voxel_size=1, fat_range=(-200, -25), cartilage_range=(200, 500), cortical_bone_range=(600, 3000), soft_tissue_range=(20, 200), obj_name_dict=None):
        self.hu_range = hu_range
        self.fat_range = fat_range
        self.cartilage_range = cartilage_range
        self.cortical_bone_range = cortical_bone_range
        self.soft_tissue_range = soft_tissue_range
        self.voxel_size = voxel_size
        
        self.result = {
            'meta':{
                    'hu_channel': 0,
                    'fat_skl_channel': 1,
                    'tissue_channel': 2,
                    'spine_marker_channel':None,

                    'hor_start_idx': 0, # 总是0
                    'sag_start_idx': -1,
                    'cor_start_idx': -1,
                    },
            'hor_slices':[],
            'sag_slices':[],
            'cor_slices':[],
        }
        
        if isinstance(obj_name_dict, dict):
            self.obj_name_dict = obj_name_dict
        else:
            self.obj_name_dict = {
                'C1':'Atlas_(C1).obj',
                'C2':'Axis_(C2).obj',
                'C3':'Vertebra_C3.obj',
                'C4':'Vertebra_C4.obj',
                'C5':'Vertebra_C5.obj',
                'C6':'Vertebra_C6.obj',
                'C7':'Vertebra_C7.obj',
                'T1':'Vertebra_T1.obj',
                'T2':'Vertebra_T2.obj',
                'T3':'Vertebra_T3.obj',
                'T4':'Vertebra_T4.obj',
                'T5':'Vertebra_T5.obj',
                'T6':'Vertebra_T6.obj',
                'T7':'Vertebra_T7.obj',
                'T8':'Vertebra_T8.obj',
                'T9':'Vertebra_T9.obj',
                'T10':'Vertebra_T10.obj',
                'T11':'Vertebra_T11.obj',
                'T12':'Vertebra_T12.obj',
                'L1':'Vertebra_L1.obj',
                'L2':'Vertebra_L2.obj',
                'L3':'Vertebra_L3.obj',
                'L4':'Vertebra_L4.obj',
                'L5':'Vertebra_L5.obj',
                'S0':'Sacrum.obj' # 骶骨S1 与 尾骨S2 合并为 S0
            }

    def load_dict(self, data):
        # 检查key 是否和 self.result一致
        for key in self.result.keys():
            if key not in data.keys():
                raise ValueError(f'Key {key} not in self.result')
        self.result = data
        
        return self.result
    
    def load_npz(self, ct_npz_path, meta_only=False):
        # 加载数据并添加到缓存
        npz_data = np.load(ct_npz_path, allow_pickle=True)
        data = {
            'meta': npz_data['meta'].item(),
        }
        if meta_only:
            return data
        data['hor_slices'] = npz_data['hor_slices'][:]
        data['sag_slices'] = npz_data['sag_slices'][:]
        data['cor_slices'] = npz_data['cor_slices'][:]

        return self.load_dict(data)
    
    def obj_to_voxel(self, mesh, array_shape, spacing, needs_extension):
        """
        使用 open3d 将 OBJ 网格转换为体素网格，并在每个体素周围根据 voxel_size 填充。

        Args:
        - mesh: 读取的 OBJ 网格。
        - array_shape: 目标 NIfTI 图像的形状。
        - spacing: NIfTI 图像的空间间隔。

        Returns:
        - voxel_grid: 转换和填充后的体素网格，作为 numpy 数组。
        - ext_voxel_grid: 扩展的体素网格，如果需要扩展的话。
        - origin_in_ext: 原点在扩展体素网格中的新位置。
        """
        pcd = mesh.sample_points_uniformly(number_of_points=5000)
        points = np.asarray(pcd.points)
        points[:, 0] = -points[:, 0]  # X轴翻转
        points = points[:, [1, 2, 0]]  # 转化到 Y, Z, X
        voxel_coords = np.round(points / spacing).astype(int)
        
        # 计算可能的最小和最大坐标
        min_coords = np.min(voxel_coords, axis=0)
        max_coords = np.max(voxel_coords, axis=0)
        
        voxel_grid = np.zeros(array_shape, dtype=np.uint8)
        origin_in_ext = np.array([0, 0, 0])
        
        # 确定是否需要扩展矩阵
        if needs_extension:
            needs_extension = np.any(min_coords < 0) or np.any(max_coords >= np.array(array_shape))
            min_coords = np.minimum(min_coords, 0)
            max_coords -= min_coords - 1
            
            # 计算扩展矩阵的新尺寸
            voxel_grid = np.zeros(max_coords, dtype=np.uint8)
            origin_in_ext = -min_coords  # 原点在扩展矩阵中的新位置
            
        offsets = np.arange(-self.voxel_size, self.voxel_size + 1)
        offset_arr = np.array(np.meshgrid(offsets, offsets, offsets)).T.reshape(-1,3)

        for voxel_coord in voxel_coords:
            if needs_extension:
                coords = voxel_coord - min_coords + offset_arr
                valid_coords = np.all((coords >= 0) & (coords < max_coords), axis=1)
                voxel_grid[tuple(coords[valid_coords].T)] = 255
            else:
                coords = voxel_coord + offset_arr
                valid_coords = np.all((coords >= 0) & (coords < array_shape), axis=1)
                voxel_grid[tuple(coords[valid_coords].T)] = 255
            return voxel_grid, origin_in_ext
    
    def load_obj_pack(self, obj_input_dir, nifti_shape, spacing, needs_extension):
        
        # 初始化最终扩展矩阵和原点坐标
        final_ext_shape = np.array(nifti_shape)
        final_origin_in_ext = np.array([0, 0, 0])
        
        # 初始化扩展数组为 None，它将在需要时创建
        if needs_extension:
            rst_array = None
        else:
            rst_array = np.zeros(nifti_shape, dtype=np.float16)
        
        # 遍历所有OBJ模型
        obj_paths = [os.path.join(obj_input_dir, obj_name) for obj_name in self.obj_name_dict.values()]
        index_values = [ c for c in range(1, len(obj_paths)+1) ]
        self.result['meta']['spine_marker'] = {tag: index for tag, index in zip(self.obj_name_dict.keys(), index_values)}
        # 为每个OBJ模型创建体素网格
        pbar = tqdm(zip(obj_paths, index_values), total=len(obj_paths))
        for obj_path, index_value in pbar:
            mesh = self.read_obj_model(obj_path)
            voxel_grid, ext_origin = self.obj_to_voxel(mesh, nifti_shape, spacing, needs_extension)
            
            if not needs_extension:
                # 使用spine_marker通道表示OBJ模型
                mask = voxel_grid > 0
                rst_array[mask] = index_value
            else:
                if ext_origin is not None:
                    
                    if rst_array is None: # 如果这是第一个需要扩展的模型，初始化 rst_array
                        rst_array = np.zeros_like(voxel_grid, dtype=np.float16)
                        final_ext_shape = voxel_grid.shape
                        final_origin_in_ext = ext_origin
                        
                        mask = voxel_grid > 0
                        rst_array[mask] = index_value
                        
                    else: # 如果已经初始化了 rst_array
                        # 计算新的扩展矩阵的最小和最大坐标
                        new_min_coords = np.minimum(final_origin_in_ext, ext_origin)
                        new_max_coords = np.maximum(final_origin_in_ext + final_ext_shape, ext_origin + voxel_grid.shape)
                        new_shape = new_max_coords - new_min_coords
                        new_origin_in_ext = new_min_coords
                        
                        # 创建新的扩展矩阵
                        new_ext_array = np.zeros(new_shape, dtype=np.float16)
                        offset_a = new_origin_in_ext - final_origin_in_ext
                        
                        new_ext_array[offset_a[0]:offset_a[0]+final_ext_shape[0],
                                    offset_a[1]:offset_a[1]+final_ext_shape[1],
                                    offset_a[2]:offset_a[2]+final_ext_shape[2]] = rst_array
                        
                        rst_array = new_ext_array
                        offset_g = new_origin_in_ext - ext_origin
                        mask = voxel_grid > 0
                        
                        rst_array[offset_g[0]:offset_g[0]+voxel_grid.shape[0],
                                offset_g[1]:offset_g[1]+voxel_grid.shape[1],
                                offset_g[2]:offset_g[2]+voxel_grid.shape[2]][mask] = index_value
                        
                        final_ext_shape = new_shape
                        final_origin_in_ext = new_origin_in_ext

        self.result['meta']['spine_marker_channel'] = 3
        
        return rst_array, final_origin_in_ext
        
    def __call__(self, nifti_path, obj_input_dir=None,  ex_meta_info=None, needs_extension=False):
        if ex_meta_info is not None:
            for key, val in ex_meta_info.items():
                self.result['meta'][key] = val
                
        # 读取NIfTI图像
        nifti_img = sitk.ReadImage(nifti_path)
        nifti_array = sitk.GetArrayFromImage(nifti_img)
        spacing = np.array(nifti_img.GetSpacing())

        fdata_array = np.zeros((*nifti_array.shape, 4), dtype=np.float16)
        o_ext = np.array([0, 0, 0])
        if obj_input_dir is not None:
            # 读取OBJ模型
            obj_array, o_ext = self.load_obj_pack(obj_input_dir, nifti_array.shape, spacing, needs_extension)
            fdata_array = np.zeros((*obj_array.shape, 4), dtype=np.float16)
        
        sm_ch = self.result['meta']['spine_marker_channel']
        # 准备结果图像的数组
        
        if sm_ch is not None:
            fdata_array[:,:,:,sm_ch] = obj_array
            
        # 将NIfTI图像数据进行归一化
        array_normalized = self.normalize_hu(nifti_array)
        array_fat_skl = self.normalize_fat_skl(nifti_array)
        array_tissue = self.normalize_tissue(nifti_array)
        
        # 横断面切片(Hor)
        depth = array_normalized.shape[0]
        hor_start_idx = 0
        hor_end_idx = -1
        pbar = tqdm(range(depth))
        for i in pbar:
            fdata_slice = fdata_array[i+o_ext[0], o_ext[1]:, o_ext[2]:, :]
            if sm_ch is not None:
                if fdata_slice[:,:,sm_ch].max() == 0: # 跳过空白切片
                    if hor_end_idx < 0:
                        hor_start_idx += 1
                        continue
                    else:
                        break
            
            fdata_slice[:, :, self.result['meta']['hu_channel']] = array_normalized[i, :, :]
            fdata_slice[:, :, self.result['meta']['fat_skl_channel']] = array_fat_skl[i, :, :]
            fdata_slice[:, :, self.result['meta']['tissue_channel']] = array_tissue[i, :, :]

            self.result['hor_slices'].append(fdata_slice.astype(np.float16))
            hor_end_idx = i
            pbar.set_description(f'Hor: {i-hor_start_idx:04d}')

        # 矢状面切片(Sag)
        width = array_normalized.shape[2]
        pbar = tqdm(range(width))
        for i in pbar:
            fdata_slice = fdata_array[hor_end_idx+o_ext[0]:hor_start_idx+o_ext[0]:-1, o_ext[1]:, o_ext[2]+i, :]
            if sm_ch is not None:
                if fdata_slice[:,:,sm_ch].max() == 0: # 跳过空白切片
                    continue

            if self.result['meta']['sag_start_idx'] == -1:
                self.result['meta']['sag_start_idx'] = i

            self.result['sag_slices'].append(fdata_slice.astype(np.float16))
            
            pbar.set_description(f'Sag: {i}')

        # 冠状面切片(Cor)    
        height = array_normalized.shape[1]
        pbar = tqdm(range(height))
        for i in pbar:
            fdata_slice = fdata_array[hor_end_idx+o_ext[0]:hor_start_idx+o_ext[0]:-1, o_ext[1]+i, o_ext[2]:, :]
            if sm_ch is not None:
                if fdata_slice[:,:,sm_ch].max() == 0: # 跳过空白切片
                    continue

            if self.result['meta']['cor_start_idx'] == -1:
                self.result['meta']['cor_start_idx'] = i

            self.result['cor_slices'].append(fdata_slice.astype(np.float16))
            
            pbar.set_description(f'Cor: {i}')
            
        # 更新meta信息
        self.result['meta']['hor_slices_num'] = len(self.result['hor_slices'])
        self.result['meta']['sag_slices_num'] = len(self.result['sag_slices'])
        self.result['meta']['cor_slices_num'] = len(self.result['cor_slices'])
        
        return self.result
        
    def normalize_hu(self, hu_image):
        """
        将HU值归一化到-1到1的浮点数范围。
        """
        print('Before normalize:', hu_image.min(), hu_image.max())
        # 将HU值从[hu_min, hu_max]范围映射到[-1, 1]
        hu_range = self.hu_range[1] - self.hu_range[0]
        normalized = 2 * ((hu_image - self.hu_range[0]) / hu_range) - 1
        print('After normalize:', normalized.min(), normalized.max())
        # 截断超过-1到1的值
        normalized[normalized < -1] = -1
        normalized[normalized > 1] = 1
        
        return normalized

    def normalize_fat_skl(self, hu_image):
        normalized = np.full(hu_image.shape, -1.0)  # 初始化为-1
        
        # 脂肪
        fat_mask = (hu_image >= self.fat_range[0]) & (hu_image <= self.fat_range[1])
        normalized[fat_mask] = np.interp(hu_image[fat_mask], [self.fat_range[0], self.fat_range[1]], [-0.9, -0.5])
        
        # 软骨
        cartilage_mask = (hu_image >= self.cartilage_range[0]) & (hu_image <= self.cartilage_range[1])
        normalized[cartilage_mask] = np.interp(hu_image[cartilage_mask], [self.cartilage_range[0], self.cartilage_range[1]], [0.1, 0.4])
        
        # 皮质骨
        cortical_bone_mask = (hu_image >= self.cortical_bone_range[0])
        normalized[cortical_bone_mask] = np.interp(hu_image[cortical_bone_mask], [self.cortical_bone_range[0], self.cortical_bone_range[1]], [0.5, 1.0])
        
        return normalized

    def normalize_tissue(self, hu_image):
        normalized = np.full(hu_image.shape, -1.0)  # 初始化为-1
        
        # 软组织
        soft_tissue_mask = (hu_image >= self.soft_tissue_range[0]) & (hu_image <= self.soft_tissue_range[1])
        normalized[soft_tissue_mask] = np.interp(hu_image[soft_tissue_mask], [self.soft_tissue_range[0], self.soft_tissue_range[1]], [-0.9, 0.9])
        
        # 皮质骨
        cortical_bone_mask = (hu_image >= self.cortical_bone_range[0])
        normalized[cortical_bone_mask] = 1.0  # 皮质骨设置为1
        
        return normalized

    def read_obj_model(self, obj_path):
        mesh = o3d.io.read_triangle_mesh(obj_path)
        mesh.scale(1000, center=[0,0,0])  # 将模型缩放到与NIfTI数据相匹配
        return mesh
    
    def interpolate_frames(self, all_slices, slice_size, cat_prev_frame):
        # 确保 slice_size 是整数
        assert isinstance(slice_size, int), "slice_size must be an integer"
        # 获取输入的形状
        _, _, h, w = all_slices.shape
        # 计算缩放因子，以最长边为准
        scale_factor = slice_size / max(h, w)
        # 如果 cat_prev_frame 为 True，有 50% 的概率在 scale_factor 基础上乘以一个 [0.5~1] 的随机缩小因子
        if cat_prev_frame and random.random() < 0.5:
            scale_factor *= random.uniform(0.5, 1)
        # 计算新的高度和宽度
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        # 使用双线性插值进行缩放
        frames = F.interpolate(all_slices, size=(new_h, new_w), mode='bilinear', align_corners=False)
        # 如果短边小于 slice_size，将图像放在新的 slice_size * slice_size 的张量中间
        if new_h < slice_size or new_w < slice_size:
            # 创建新的张量
            new_frames = torch.ones((all_slices.shape[0], all_slices.shape[1], slice_size, slice_size), device=all_slices.device) * -1
            # 计算开始的索引
            start_h = (slice_size - new_h) // 2
            start_w = (slice_size - new_w) // 2
            # 将缩放后的图像放在新的张量中间
            new_frames[:, :, start_h:start_h+new_h, start_w:start_w+new_w] = frames
            # 更新 frames
            frames = new_frames
        # 如果 cat_prev_frame 为 True，随机在 frames[:-1] 中裁剪一些方块并设置为 0
        # 如果 cat_prev_frame 为 True，随机在 frames[:-1] 中裁剪一些方块并设置为 0
        if cat_prev_frame:
            # 计算裁剪的方块数量
            num_blocks = random.randint(0, 5)
            for _ in range(num_blocks):
                # 随机选择一个方块
                block_h = random.randint(0, slice_size // 2)
                block_w = random.randint(0, slice_size // 2)
                # 设置这个方块为 0
                if random.random() < 0.5:
                    # 随机选择一个起点
                    end_h = random.randint(block_h, slice_size // 2)
                    end_w = random.randint(block_w, slice_size // 2)
                    frames[:-1, :3, block_h:end_h, block_w:end_w] = -1
                else:
                # 随机选择一个边缘并裁剪一定的宽度
                    edge = random.choice(['top', 'bottom', 'left', 'right'])
                    if edge == 'top':
                        frames[:-1, :3, :block_h, :] = -1
                    elif edge == 'bottom':
                        frames[:-1, :3, -block_h:, :] = -1
                    elif edge == 'left':
                        frames[:-1, :3, :, :block_w] = -1
                    elif edge == 'right':
                        frames[:-1, :3, :, -block_w:] = -1

        return frames
        
    
    def to_frame(self, plane, start_idx, ct=None, slice_size=None, sample_num=3, positive_order=True, randomize_sentence=True, cat_prev_frame=True):
        if ct is None:
            ct = self.result
        
        slices = ct[plane]
        sm_ch = ct['meta']['spine_marker_channel']
        properties = {
        'plane': plane[:-7], 
        'positive_order': positive_order, 
        'spines': '',
        'original_sizes': slices[0].shape[:2]
        }
        if 'age' in ct['meta']:
            properties['age'] = ct['meta']['age']
        if 'gender' in ct['meta']:
            properties['gender'] = ct['meta']['gender']
        
        # 假设slices是一个NumPy数组，形状为[切片数量, 高度, 宽度, 通道数]
        all_slices = torch.from_numpy(slices[start_idx:start_idx+sample_num]).to(torch.float32)  # 转换为torch张量
        all_slices = all_slices.permute(0, 3, 1, 2)  # 重排维度为[B, C, H, W]
        
        frames = self.interpolate_frames(all_slices, slice_size, cat_prev_frame)
        
        # 获取存在的脊柱tag
        obj_tags = []
        for obj_tag, index_value in ct['meta']['spine_marker'].items():
            if index_value in frames[-1,sm_ch]:
                obj_tags.append(obj_tag)
                properties['spines'] = '|'.join(obj_tags)
        # sm_ch 归一化
        frames[:,sm_ch] = frames[:,sm_ch] / 12.5 - 1
        properties['sentence'] = self.dict_to_sentence(properties, randomize=randomize_sentence)
        
        return frames, properties