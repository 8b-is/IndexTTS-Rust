import requests
import zipfile
import os
import argparse

def get_confirm_token(response):
    """
    从响应中检查是否存在下载确认令牌（cookie）

    Args:
        response (requests.Response): 响应对象

    Returns:
        str: 确认令牌的值（如果存在），否则为None
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'): # 确认令牌的cookie通常以这个开头
            return value
    return None

def save_response_content(response, destination, chunk_size=32768):
    """
    以流式方式将响应内容写入文件，支持大文件下载。

    Args:
        response (requests.Response): 流式响应对象
        destination (str): 本地保存路径
        chunk_size (int, optional): 每次迭代写入的块大小. Defaults to 32768.
    """
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk: # 过滤掉保持连接的空白块
                f.write(chunk)

def download_model_from_modelscope(destination,hf_cache_dir):
    print(f"[ModelScope] Downloading models to {destination},model cache dir={hf_cache_dir}")
    from modelscope import snapshot_download
    snapshot_download("IndexTeam/IndexTTS-2", local_dir=destination)
    snapshot_download("amphion/MaskGCT", local_dir=os.path.join(hf_cache_dir,"models--amphion--MaskGCT"))
    snapshot_download("facebook/w2v-bert-2.0",local_dir=os.path.join(hf_cache_dir,"models--facebook--w2v-bert-2.0"))
    snapshot_download("nv-community/bigvgan_v2_22khz_80band_256x",local_dir=os.path.join(hf_cache_dir,"models--nvidia--bigvgan_v2_22khz_80band_256x"))
    snapshot_download("iic/speech_campplus_sv_zh-cn_16k-common",local_dir=os.path.join(hf_cache_dir,"models--funasr--campplus"))

def download_model_from_huggingface(destination,hf_cache_dir):
    print(f"[HuggingFace] Downloading models to {destination},model cache dir={hf_cache_dir}")
    from huggingface_hub import snapshot_download
    snapshot_download("IndexTeam/IndexTTS-2", local_dir=destination)
    # snapshot_download("amphion/MaskGCT", local_dir=os.path.join(hf_cache_dir,"models--amphion--MaskGCT"))
    # snapshot_download("facebook/w2v-bert-2.0",local_dir=os.path.join(hf_cache_dir,"models--facebook--w2v-bert-2.0"))
    # snapshot_download("nvidia/bigvgan_v2_22khz_80band_256x",local_dir=os.path.join(hf_cache_dir, "models--nvidia--bigvgan_v2_22khz_80band_256x"))
    # snapshot_download("funasr/campplus",local_dir=os.path.join(hf_cache_dir,"models--funasr--campplus"))

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models and example files")
    parser.add_argument('-s','--model_source', choices=['modelscope', 'huggingface'], default=None, help='Model source')
    args = parser.parse_args()

    if args.model_source:
        if args.model_source == 'modelscope':
            download_model_from_modelscope("checkpoints",os.path.join("checkpoints","hf_cache"))
        elif args.model_source == 'huggingface':
            download_model_from_huggingface("checkpoints",os.path.join("checkpoints","hf_cache"))
