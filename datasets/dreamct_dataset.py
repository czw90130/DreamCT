from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import random
import time
import gc

'''
读取CT切片数据
{
    'meta':{
                'hu_channel': 0, # HU通道
                'fat_skl_channel': 1, # 脂肪骨骼通道
                'tissue_channel': 2, # 软组织通道
                'spine_marker_channel':3, # 脊柱标记通道

                'hor_start_idx': 0, # 总是0
                'sag_start_idx': 120, # 从有脊柱标记的切片开始
                'cor_start_idx': 130, # 从有脊柱标记的切片开始
                'spine_marker': ['C1':1, 'C2':2, 'C3':3, 'C4':4, ...] # 脊柱标记
                },
        'hor_slices':[], # 水平(Horizontal Plane)切片
        'sag_slices':[], # 矢状(Sagittal Plane)切片
        'cor_slices':[], # 冠状(Coronal Plane)切片
}
'''

class CTdataProcessor:
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
        
        # 确保至少有一个部分填充
        filled_templates = [t for t in templates if any(parts.values())]
        if not filled_templates:
            return "Patient details are not fully specified."

        # 随机选择一个模板并返回
        return random.choice(templates).replace("  ", " ").replace(",,", ",").strip()
    
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

        return data

    def __call__(self, ct_npz_path, plane, start_idx, slice_size=None, sample_num=3, positive_order=True, randomize_sentence=True):
        return self.to_frame(ct_npz_path, plane, start_idx, slice_size, sample_num, positive_order, randomize_sentence)
    
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
        
    
    def to_frame(self, ct, plane, start_idx, slice_size=None, sample_num=3, positive_order=True, randomize_sentence=True, cat_prev_frame=True):
        if isinstance(ct, str):
            ct = self.load_npz(ct)
        slices = ct[plane]
        sm_ch = ct['meta']['spine_marker_channel']
        
        properties = {
        'plane': plane[:-7], 
        'positive_order': positive_order, 
        'spines': '',
        'age': ct['meta']['age'],
        'gender': ct['meta']['gender'],
        'original_sizes': slices[0].shape[:2]
        }
        
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
        

class CTFramesDataset(Dataset, CTdataProcessor):
    '''
    读取CT切片数据
    '''
    def __init__(
        self,
        ct_data_root,
        slice_size=512, # 切片重采样大小
        frame_num=3+1, # 读取的切片数(3帧预测1帧)
        skip_rate= 0.5, # 跳过的概率
        train=True,
        cache_size=4,  # 缓存大小
    ):
        self.ct_data_root = Path(ct_data_root)
        self.slice_size = slice_size
        self.train = train
        self.frame_num = frame_num
        self.cache_size = cache_size
        self.cache = OrderedDict()  # 使用OrderedDict作为缓存

        ct_npzs = list(self.ct_data_root.glob('*.npz'))
        self.data_idx_buffer = []

        pbar = tqdm(ct_npzs)
        for ct_npz in pbar:
            pbar.set_description(f"Loading {ct_npz}")
            pbar.set_postfix_str(f" {len(self.data_idx_buffer)}")
            
            sub_idx_buffer = []
            ct = self.load_data(ct_npz, meta_only=True)

            for plane in ['hor_slices', 'sag_slices', 'cor_slices']:
                try:
                    end_idx = ct['meta'][plane+'_num']-frame_num
                except KeyError:
                    end_idx = ct[plane].shape[0]-frame_num
                for i in range(0, end_idx):
                    if random.random() < skip_rate:
                        continue
                    sub_idx_buffer.append((ct_npz, plane, i))
            
            self.data_idx_buffer.extend(sub_idx_buffer)
                
    def load_data(self, ct_npz_path, meta_only=False):
        if ct_npz_path in self.cache:
            # 如果数据已经在缓存中，直接返回
            return self.cache[ct_npz_path]
        else:
            # 加载数据并添加到缓存
            data = self.load_npz(ct_npz_path, meta_only)
            if meta_only:
                return data
            self.cache[ct_npz_path] = data
            # 如果缓存超过了指定大小，移除最旧的条目
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)  # FIFO order
                gc.collect()
            return data
                
    def __len__(self):
        return len(self.data_idx_buffer)
    
    def __getitem__(self, idx):
        ct_npz, plane, start_idx = self.data_idx_buffer[idx]
        ct = self.load_data(ct_npz)
        
        return self.to_frame(ct, plane, start_idx, slice_size=self.slice_size, sample_num=self.frame_num, positive_order=random.choice([True, False]))
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import time
    # Initialize the dataset
    dataset = CTFramesDataset(ct_data_root="D:\\CT\\scripts\\HuColorful\\dataset_thorax_abdomen_pelvis")
    
    # Create a DataLoader
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)  # Adjust num_workers as per your system

    # Print the length of the dataset
    print(f"Length of dataset: {len(dataset)}")
    
    total_time = 0  # Initialize total time for accessing data batches
    prev_time = time.time()  # Start time measurement
    # Iterate over the DataLoader
    for i, data in enumerate(loader):
        # Processing data here
        # Since DataLoader batches data, `data` here is a batch
        access_time = time.time()-prev_time
        total_time += access_time  # Update total time
        
        # Print batch information
        print(f"========> Batch: {i} <========")
        # Example for accessing and printing data, adjust according to your data structure
        print(data[0].shape)
        print(data[1]['positive_order'])
        print(data[1]['sentence'])
        
        # Calculate and display average access time up to the current point
        average_time = total_time / (i + 1)
        print(f"Average access time up to batch {i}: {average_time:.4f} seconds")
        prev_time = time.time()