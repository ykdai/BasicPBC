import argparse
import os
from os import path as osp

from basicsr.utils.download_util import load_file_from_url


def download_pretrained_models(file_name, file_url):
    save_path_root = f'./ckpt/'
    save_path = f'{file_name}'
    os.makedirs(save_path_root, exist_ok=True)

    save_path = load_file_from_url(url=file_url, model_dir=save_path_root, progress=True, file_name=save_path)


if __name__ == '__main__':

    method='basicpbc.pth'
    file_path = 'https://github.com/ykdai/BasicPBC/releases/download/v0.1.0/basicpbc.pth'
    download_pretrained_models(method, file_path)

    method='basicpbc_light.pth'
    file_path = 'https://github.com/ykdai/BasicPBC/releases/download/v0.1.0/basicpbc_light.pth'
    download_pretrained_models(method, file_path)

