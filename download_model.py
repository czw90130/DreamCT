from huggingface_hub import snapshot_download
# 设置默认模型路径和本地缓存路径
DEFAULT_MODEL = "stabilityai/stable-diffusion-2"
cache_dir = "./huggingface_model/"    # diffusers的本地缓存路径
snapshot_download(DEFAULT_MODEL, local_dir=cache_dir)