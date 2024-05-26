import argparse
import os
from os import path as osp

from basicsr.utils.download_util import load_file_from_url


def download_pretrained_models(method, file_urls):
    save_path_root = f'./ckpt/{method}'
    os.makedirs(save_path_root, exist_ok=True)

    for file_name, file_url in file_urls.items():
        save_path = load_file_from_url(url=file_url, model_dir=save_path_root, progress=True, file_name=file_name)


if __name__ == '__main__':

    method='basicpbc.pth'
    file_path = 'https://github.com/ykdai/BasicPBC/releases/download/v0.1.0/basicpbc.pth'
    download_pretrained_models(method, file_path)

    method='basicpbc_light.pth'
    file_path = 'https://github.com/ykdai/BasicPBC/releases/download/v0.1.0/basicpbc_light.pth'
    download_pretrained_models(method, file_path)

