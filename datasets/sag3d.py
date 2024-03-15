import os
import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
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
                if v != '':
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
    
class Sag3DEncoder(SentenceBuilder):
    organ_color = {
        'skin': [228, 200, 166],
        'bone': [255, 255, 255],
        'C1': [206, 210, 235],
        'C2': [221, 200, 220],
        'C3': [251, 190, 190],
        'C4': [255, 190, 190],
        'C5': [255, 190, 190],
        'C6': [255, 190, 190],
        'C7': [255, 190, 190],
        'T1': [205, 255, 190],
        'T2': [190, 255, 190],
        'T3': [190, 255, 190],
        'T4': [190, 255, 190],
        'T5': [190, 255, 190],
        'T6': [190, 255, 190],
        'T7': [190, 255, 190],
        'T8': [190, 255, 190],
        'T9': [190, 255, 190],
        'T10': [190, 255, 190],
        'T11': [190, 255, 190],
        'T12': [190, 255, 190],
        'L1': [240, 190, 235],
        'L2': [230, 190, 250],
        'L3': [210, 190, 255],
        'L4': [190, 190, 255],
        'L5': [190, 190, 255],
        'subcutaneous_fat': [255, 255, 0],
        'muscle': [255, 0, 0],
        'torso_fat': [139, 69, 19],
        'chest_organs': [0, 0, 255],
        'belly_organs': [0, 255, 0],
        'brain':[128, 0, 128],
    }

    def __init__(self, hu_range=(-1000, 3000)):
        self.hu_range = hu_range
        self.mat = None
        self.info = None

    def load_npz(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.info = data['info'].item()
        self.mat = torch.from_numpy(data['data'])
        return self.mat, self.info

    def interpolate_slice(self, slice_data, target_size, crop):
        # 获取输入的形状
        _, h, w, _ = slice_data.shape
        if crop or isinstance(crop, int):
            # 以最短边为准，选择随机起点裁剪到正方形
            if h < w:
                if not isinstance(crop, int):
                    start = random.randint(0, w - h)
                else:
                    start = crop
                if start + h > w:
                    start = w - h
                slice_data = slice_data[:, :, start:start+h, :]
            else:
                if not isinstance(crop, int):
                    start = random.randint(0, h - w)
                else:
                    start = crop
                if start + w > h:
                    start = h - w
                slice_data = slice_data[:, start:start+w, :, :]
            slice_data = slice_data.permute(0, 3, 1, 2).to(torch.float32)
            # 使用双线性插值进行缩放
            slice_data = F.interpolate(slice_data, size=(target_size, target_size), mode='bilinear', align_corners=False)
        else:
            # 计算缩放因子，以最长边为准
            scale_factor = target_size / max(h, w)
            # 计算新的高度和宽度
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            slice_data = slice_data.permute(0, 3, 1, 2).to(torch.float32)
            # 使用双线性插值进行缩放
            slice_data = F.interpolate(slice_data, size=(new_h, new_w), mode='bilinear', align_corners=False)
            # 如果短边小于 slice_size，将图像放在新的 slice_size * slice_size 的张量中间
            if new_h < target_size or new_w < target_size:
                pad_h = target_size - new_h
                pad_w = target_size - new_w
                slice_data = F.pad(slice_data, (pad_w//2, pad_w//2, pad_h//2, pad_h//2), mode='constant', value=-1)

        return slice_data.permute(0, 2, 3, 1)[0]

    def normalize(self, data, min_val=-1, max_val=1, mode='color'):
        if mode == 'color':
            data_min = 0
            data_max = 255
        elif mode == 'ct':
            data_min, data_max = self.hu_range
        else:
            data_min = torch.min(data)
            data_max = torch.max(data)
        rst = (max_val - min_val) * (data - data_min) / (data_max - data_min) + min_val
        return rst.clip(min_val, max_val)

    def to_frames(self, axis, start_index, data=None, crop=None, target_size=256, num_frames=32):
        if data is None:
            data = self.mat
        if data is None:
            raise ValueError("No data loaded")
        
        # assert 0 < start_index <= data.shape[axis] - num_frames, f"Invalid start index for axis {axis}: start_index:{start_index}, data_shape:{data.shape[axis]}"

        if axis == 0:
            clip_data = data[start_index:start_index+num_frames,:,:,:]
        elif axis == 1:
            clip_data = data[:,start_index:start_index+num_frames,:,:]
        else:
            clip_data = data[:,:,start_index:start_index+num_frames,:]

        frames = []
        for i in range(num_frames):
            if axis == 0:
                slice_data = clip_data[i:i+1,:,:,:]
            elif axis == 1:
                slice_data = clip_data[:,i:i+1,:,:].permute(1, 0, 2, 3)
            else:
                slice_data = clip_data[:,:,i:i+1,:].permute(2, 0, 1, 3)
            slice_data = self.interpolate_slice(slice_data, target_size, crop)
            rgb_data = self.normalize(slice_data[...,:3])

            ct_data = self.normalize(slice_data[...,3:4], mode='ct')
            ct_data = torch.cat([ct_data, ct_data, ct_data], dim=-1)

            smpl_data = self.normalize(slice_data[...,4:])

            # frames.append({'sagement':rgb_data, 'ct':ct_data, 'smpl':smpl_data})
            frames.append(torch.cat([rgb_data, ct_data, smpl_data], dim=-1))
        frames = torch.stack(frames, dim=0)
        return frames
    
class Sag3DDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, target_size=256, num_frames=32, axes=[0, 1, 2], num_samples_per_npz=100, output_mode='random', output_with_info=True):
        self.root_dir = root_dir
        self.target_size = target_size
        self.num_frames = num_frames
        self.axes = axes
        self.num_samples_per_npz = num_samples_per_npz
        self.output_with_info = output_with_info
        self.output_mode = output_mode
        self.encoder = Sag3DEncoder()

        self.npz_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".npz")]
        self.data_list = self._generate_data_list()

    def _is_all_black(self, mat, axis, start_index):
        if axis == 0:
            return torch.all(mat[start_index:start_index+self.num_frames, :, :, :3] == -1)
        elif axis == 1:
            return torch.all(mat[:, start_index:start_index+self.num_frames, :, :3] == -1)
        else:
            return torch.all(mat[:, :, start_index:start_index+self.num_frames, :3] == -1)
    
    def _generate_data_list(self):
        data_list = []
        for npz_file in self.npz_files:
            mat,_  = self.encoder.load_npz(npz_file)
            for _ in range(self.num_samples_per_npz):
                axis = random.choice(self.axes)
                max_start_index = mat.shape[axis] - self.num_frames
                if max_start_index < 1:
                    raise ValueError(f"Insufficient frames in {npz_file}, axis: {axis}")
                
                start_index = random.randint(0, max_start_index)
                while self._is_all_black(mat, axis, start_index):
                    start_index = random.randint(0, max_start_index)

                crop = random.choice([True, True, True, False])
                data_list.append({"npz_file": npz_file, "axis": axis, "start_index": start_index, "crop": crop})
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_info = self.data_list[idx]
        npz_file, axis, start_index, crop = data_info["npz_file"], data_info["axis"], data_info["start_index"], data_info["crop"]

        mat, info = self.encoder.load_npz(npz_file)
        frames = self.encoder.to_frames(axis, start_index, mat, crop, self.target_size, self.num_frames)
        if self.output_mode == 'all':
            pass
        elif self.output_mode == 'segment':
            frames = frames[..., :3]
        elif self.output_mode == 'ctvalue':
            frames = frames[..., 3:6]
        elif self.output_mode == 'smpl':
            frames = frames[..., 6:]
        else:
            frames = random.choice([frames[..., :3], frames[..., 3:6], frames[..., 6:]])
        output = {'data':frames.permute(3, 0, 1, 2)}
        if self.output_with_info:
            output['info'] = info
        return output
    
if __name__ == "__main__":
    import imageio

    npz_path = 'testdata/'  # 替换为您的npz文件路径
    out_gif_path = 'testdata/result/output.gif'
    
    dataset = Sag3DDataset(npz_path)
    print('dataset:', len(dataset))
    frames = dataset[0]

    gif_frames = []
    # 逐帧处理数据
    for frame in frames:
        frame = frame.numpy()
        # # 将归一化的数据缩放到[0, 255]的范围内,并转换为uint8类型
        # rgb_data = ((frame[..., :3] + 1) * 127.5).astype(np.uint8)
        # ct_data = ((frame[..., 3:6] + 1) * 127.5).astype(np.uint8)
        # smpl_data = ((frame[..., 6:] + 1) * 127.5).astype(np.uint8)

        # # 将RGB、CT和SMPL数据水平拼接起来
        # img = np.concatenate([rgb_data, ct_data, smpl_data], axis=1)
        img = ((frame + 1) * 127.5).astype(np.uint8)

        # 将图像添加到gif_frames列表中
        gif_frames.append(img)

    # 使用imageio创建GIF图像
    imageio.mimsave(out_gif_path, gif_frames, fps=3)

        
    
        