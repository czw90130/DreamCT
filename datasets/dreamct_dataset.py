from torch.utils.data import Dataset
from pathlib import Path

from tqdm import tqdm
from collections import OrderedDict
import random
import time
import gc
from nifti_mat import NIfTIEncoder

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
class CTFramesDataset(Dataset, NIfTIEncoder):
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
        NIfTIEncoder.__init__(self)
        
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
        
        return self.to_frame(plane, start_idx, ct=ct, slice_size=self.slice_size, sample_num=self.frame_num, positive_order=random.choice([True, False]))
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import sys
    import time
    # Initialize the dataset
    dataset = CTFramesDataset(ct_data_root=sys.argv[1])

    # Print the length of the dataset
    print(f"Length of dataset: {len(dataset)}")
    
    # Create a DataLoader
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)  # Adjust num_workers as per your system
    
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