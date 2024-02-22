import os
import sys
import cv2
import open3d as o3d
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import random

def normalize_hu(hu_image, hu_min=-1000, hu_max=3000):
    """
    将HU值归一化到-1到1的浮点数范围。
    """
    print('Before normalize:', hu_image.min(), hu_image.max())
    # 将HU值从[hu_min, hu_max]范围映射到[-1, 1]
    hu_range = hu_max - hu_min
    normalized = 2 * ((hu_image - hu_min) / hu_range) - 1
    print('After normalize:', normalized.min(), normalized.max())
    # 截断超过-1到1的值
    normalized[normalized < -1] = -1
    normalized[normalized > 1] = 1
    
    return normalized

def normalize_fat_skl(hu_image, fat_range=(-200, -25), cartilage_range=(200, 500), cortical_bone_range=(600, 3000)):
    normalized = np.full(hu_image.shape, -1.0)  # 初始化为-1
    
    # 脂肪
    fat_mask = (hu_image >= fat_range[0]) & (hu_image <= fat_range[1])
    normalized[fat_mask] = np.interp(hu_image[fat_mask], [fat_range[0], fat_range[1]], [-0.9, -0.5])
    
    # 软骨
    cartilage_mask = (hu_image >= cartilage_range[0]) & (hu_image <= cartilage_range[1])
    normalized[cartilage_mask] = np.interp(hu_image[cartilage_mask], [cartilage_range[0], cartilage_range[1]], [0.1, 0.4])
    
    # 皮质骨
    cortical_bone_mask = (hu_image >= cortical_bone_range[0])
    normalized[cortical_bone_mask] = np.interp(hu_image[cortical_bone_mask], [cortical_bone_range[0], cortical_bone_range[1]], [0.5, 1.0])
    
    return normalized

def normalize_tissue(hu_image, soft_tissue_range=(20, 200), cortical_bone_threshold=600):
    normalized = np.full(hu_image.shape, -1.0)  # 初始化为-1
    
    # 软组织
    soft_tissue_mask = (hu_image >= soft_tissue_range[0]) & (hu_image <= soft_tissue_range[1])
    normalized[soft_tissue_mask] = np.interp(hu_image[soft_tissue_mask], [soft_tissue_range[0], soft_tissue_range[1]], [-0.9, 0.9])
    
    # 皮质骨
    cortical_bone_mask = (hu_image >= cortical_bone_threshold)
    normalized[cortical_bone_mask] = 1.0  # 皮质骨设置为1
    
    return normalized

def denormalize_hu(normalized_image, hu_min=-1000, hu_max=3000):
    """
    将归一化的-1到1的浮点数值反归一化到原始HU值范围。
    """
    print('Before denormalize:', normalized_image.min(), normalized_image.max())
    # 从[-1, 1]范围反映射到[hu_min, hu_max]
    hu_range = hu_max - hu_min
    denormalized = ((normalized_image + 1) / 2) * hu_range + hu_min
    denormalized = denormalized.astype(np.float16)
    print('After denormalize:', denormalized.min(), denormalized.max())
    return denormalized

def read_obj_model(obj_path):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.scale(1000, center=[0,0,0])  # 将模型缩放到与NIfTI数据相匹配
    return mesh

def obj_to_voxel(mesh, array_shape, spacing, voxel_size=1):
    """
    使用 open3d 将 OBJ 网格转换为体素网格，并在每个体素周围根据 voxel_size 填充。

    Args:
    - mesh: 读取的 OBJ 网格。
    - array_shape: 目标 NIfTI 图像的形状。
    - spacing: NIfTI 图像的空间间隔。
    - voxel_size: 控制填充区域大小的参数。

    Returns:
    - voxel_grid: 转换和填充后的体素网格，作为 numpy 数组。
    """
    pcd = mesh.sample_points_uniformly(number_of_points=5000)
    voxel_grid_np = np.zeros(array_shape, dtype=np.uint8)

    points = np.asarray(pcd.points)
    points[:, 0] = -points[:, 0]  # X轴翻转
    points = points[:, [1, 2, 0]]  # 转化到 Y, Z, X

    voxel_coords = np.round(points / spacing).astype(int)

    offsets = np.arange(-voxel_size, voxel_size + 1)
    offset_arr = np.array(np.meshgrid(offsets, offsets, offsets)).T.reshape(-1,3)

    for voxel_coord in voxel_coords:
        coords = voxel_coord + offset_arr
        valid_coords = np.all((coords >= 0) & (coords < array_shape), axis=1)
        voxel_grid_np[tuple(coords[valid_coords].T)] = 255

    return voxel_grid_np

def combine_image(slice, meta):
    red = ((slice[:,:,meta['hu_channel']] + 1) * 127.5).astype(np.uint8)
    green = ((slice[:,:,meta['fat_skl_channel']] + 1) * 127.5).astype(np.uint8)
    blue = ((slice[:,:,meta['tissue_channel']]+ 1) * 127.5).astype(np.uint8)
    
    green[slice[:,:,meta['spine_marker_channel']] > 0] = 255
    blue[slice[:,:,meta['spine_marker_channel']] > 0] = 255
    

    return cv2.merge([blue, green, red])



def ct2dict(nifti_path, obj_input_dir, obj_name_dict, output_dir, prefix=None):
    result = {
        'meta':{
                'hu_channel': 0,
                'fat_skl_channel': 1,
                'tissue_channel': 2,
                'spine_marker_channel':3,

                'hor_start_idx': 0, # 总是0
                'sag_start_idx': -1,
                'cor_start_idx': -1,
                },
        'hor_slices':[],
        'sag_slices':[],
        'cor_slices':[],
    }
    if prefix is not None:
        for key, val in prefix.items():
            result['meta'][key] = val

    # 读取NIfTI图像
    nifti_img = sitk.ReadImage(nifti_path)
    nifti_array = sitk.GetArrayFromImage(nifti_img)
    spacing = np.array(nifti_img.GetSpacing())
    
    # 准备结果图像的数组
    fdata_array = np.zeros((*nifti_array.shape, 4), dtype=np.float16)

    # 遍历所有OBJ模型
    obj_paths = [os.path.join(obj_input_dir, obj_name) for obj_name in obj_name_dict.values()]
    index_values = [ c for c in range(1, len(obj_paths)+1) ]
    result['meta']['spine_marker'] = {tag: index for tag, index in zip(obj_name_dict.keys(), index_values)}

    sm_ch = result['meta']['spine_marker_channel']
    # 为每个OBJ模型创建体素网格
    pbar = tqdm(zip(obj_paths, index_values), total=len(obj_paths))
    for obj_path, index_value in pbar:
        mesh = read_obj_model(obj_path)
        voxel_grid = obj_to_voxel(mesh, nifti_array.shape, spacing)

        # 使用spine_marker通道表示OBJ模型
        mask = voxel_grid > 0
        fdata_array[mask, sm_ch] = index_value

    # 将NIfTI图像数据进行归一化
    array_normalized = normalize_hu(nifti_array)  # 归一化
    array_fat_skl = normalize_fat_skl(nifti_array)
    array_tissue = normalize_tissue(nifti_array)

    # 横断面切片(Hor)
    depth = array_normalized.shape[0]
    hor_start_idx = 0
    hor_end_idx = depth
    pbar = tqdm(range(depth))
    for i in pbar:
        obj_tags = []
        fdata_slice = fdata_array[i, :, :, :]
        if fdata_slice[:,:,sm_ch].max() == 0: # 跳过空白切片
            hor_start_idx += 1
            continue
        else:
            # 获取存在的tag
            for obj_tag, index_value in result['meta']['spine_marker'].items():
                if index_value in fdata_slice[:,:,sm_ch]:
                    obj_tags.append(obj_tag)
        
        fdata_slice[:, :, result['meta']['hu_channel']] = array_normalized[i, :, :]
        fdata_slice[:, :, result['meta']['fat_skl_channel']] = array_fat_skl[i, :, :]
        fdata_slice[:, :, result['meta']['tissue_channel']] = array_tissue[i, :, :]

        result['hor_slices'].append(fdata_slice.astype(np.float16))
        hor_end_idx = i
        pbar.set_description(f'Hor: {i-hor_start_idx:04d}, Tags: {obj_tags}')

    # 矢状面切片(Sag)
    width = array_normalized.shape[2]
    pbar = tqdm(range(width))
    for i in pbar:
        obj_tags = []
        fdata_slice = fdata_array[hor_end_idx:hor_start_idx:-1, :, i, :]
        if fdata_slice[:,:,sm_ch].max() == 0: # 跳过空白切片
            continue
        else:
            for obj_tag, index_value in result['meta']['spine_marker'].items():
                if index_value in fdata_slice[:,:,sm_ch]:
                    obj_tags.append(obj_tag)

        if result['meta']['sag_start_idx'] == -1:
            result['meta']['sag_start_idx'] = i

        result['sag_slices'].append(fdata_slice.astype(np.float16))
        
        pbar.set_description(f'Sag: {i}, Tags: {obj_tags}')

    # 冠状面切片(Cor)    
    height = array_normalized.shape[1]
    pbar = tqdm(range(height))
    for i in pbar:
        obj_tags = []
        fdata_slice = fdata_array[hor_end_idx:hor_start_idx:-1, i, :, :]
        if fdata_slice[:,:,sm_ch].max() == 0: # 跳过空白切片
            continue
        else:
            for obj_tag, color_value in result['meta']['spine_marker'].items():
                if color_value in fdata_slice[:,:,sm_ch]:
                    obj_tags.append(obj_tag)

        if result['meta']['cor_start_idx'] == -1:
            result['meta']['cor_start_idx'] = i

        result['cor_slices'].append(fdata_slice.astype(np.float16))
        
        pbar.set_description(f'Cor: {i}, Tags: {obj_tags}')

    # 更新meta信息
    result['meta']['hor_slices_num'] = len(result['hor_slices'])
    result['meta']['sag_slices_num'] = len(result['sag_slices'])
    result['meta']['cor_slices_num'] = len(result['cor_slices'])
    
    # 在 Hor、Sag、Cor 随机抽取一个切片保存为图片
    random_hor_slice = random.choice(result['hor_slices'])
    random_sag_slice = random.choice(result['sag_slices'])
    random_cor_slice = random.choice(result['cor_slices'])

    random_hor_img = combine_image(random_hor_slice, result['meta'])
    random_sag_img = combine_image(random_sag_slice, result['meta'])
    random_cor_img = combine_image(random_cor_slice, result['meta'])


    # 返回结果
    return result, random_hor_img, random_sag_img, random_cor_img

obj_name_dict = {
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
        'S0':'Sacrum.obj'
}

if __name__ == '__main__':
    nifti_dir = sys.argv[1] # ..\..\thorax_abdomen_pelvis\
    meta_path = sys.argv[2] # ..\..\Totalsegmentator_dataset\meta.csv
    # obj_input_dir = sys.argv[3]
    output_dir = sys.argv[3]

    # 读取 meta.csv 文件
    # image_id;age;gender;institute;study_type;split
    import pandas as pd
    meta_df = pd.read_csv(meta_path, delimiter=';')


    niftis = os.listdir(nifti_dir)
    niftis.sort()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i,nifti_name in enumerate(niftis):
        save_path = os.path.join(output_dir, f'{nifti_name}')
        # if os.path.exists(save_path+'.npy') or os.path.exists(save_path+'.npz'):
        #     print(f'File {nifti_name} exists, skip.')
        #     continue
        nifti_path = os.path.join(nifti_dir, nifti_name, 'CT.nii.gz')
        obj_input_dir = os.path.join(nifti_dir, nifti_name, 'zanatomy')

        age = meta_df[meta_df['image_id'] == nifti_name]['age'].values[0]
        gender = meta_df[meta_df['image_id'] == nifti_name]['gender'].values[0]

        print(f'Processing: {nifti_name}, Age: {age}, Gender {gender}')

        prefix = {'age': age, 'gender': gender}
    
        rst, r_hor_img, r_sag_img, r_cor_img = ct2dict(nifti_path, obj_input_dir, obj_name_dict, output_dir, prefix)

        np.savez_compressed(save_path+'.npz', **rst)
        if i%10 == 0:
            cv2.imwrite(os.path.join(output_dir, f'{nifti_name}_hor.png'), r_hor_img)
            cv2.imwrite(os.path.join(output_dir, f'{nifti_name}_sag.png'), r_sag_img)
            cv2.imwrite(os.path.join(output_dir, f'{nifti_name}_cor.png'), r_cor_img)