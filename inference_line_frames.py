import argparse
import os
import torch
import torch.utils.data as data
from glob import glob
from os import path as osp
from skimage import io, measure

from basicsr.archs.basicpbc_arch import BasicPBC
from basicsr.data.animerun_dataset import PaintBucketSegMat
from basicsr.models.pbc_model import ModelInference
from paint.lineart import LineArt
from paint.utils import dump_json, np_2_labelpng, read_seg_2_np


def generate_seg_folder(path):
    for clip in os.listdir(path):
        clip_path = osp.join(path, clip)
        line_path = osp.join(clip_path, "line")
        seg_path = osp.join(clip_path, "seg")
        os.makedirs(seg_path, exist_ok=True)
        for line in sorted(glob(osp.join(line_path, "*.png"))):
            name = osp.split(line)[-1][:-4]
            line = io.imread(line)
            lineart = LineArt(line)
            lineart.label_color_line()
            seg_np = lineart.label_img
            seg = np_2_labelpng(seg_np)
            save_path = osp.join(seg_path, name + ".png")
            io.imsave(save_path, seg, check_contrast=False)
            print(f"{save_path} created.")
        extract_color_dict(clip_path)


def extract_color_dict(clip_path):

    seg = read_seg_2_np(osp.join(clip_path, "seg", "0000.png"))
    gt = io.imread(osp.join(clip_path, "gt", "0000.png"))

    color_dict = {}
    props = measure.regionprops(seg)

    for i in range(1, seg.max() + 1):
        pos = props[i - 1].coords[0]
        index_color = gt[pos[0], pos[1], :]
        color_dict[str(i)] = index_color.tolist()

    dump_json(color_dict, osp.join(clip_path, "seg", "0000.json"))


def load_params(model_path):
    full_model = torch.load(model_path)
    if "params_ema" in full_model:
        return full_model["params_ema"]
    elif "params" in full_model:
        return full_model["params"]
    else:
        return full_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, default="dataset/test/your_data")
    args = parser.parse_args()

    path = args.folder_path
    generate_seg_folder(path)

    ckpt_path = "ckpt/basicpbc.pth"
    model = BasicPBC(
        ch_in=6,
        descriptor_dim=128,
        keypoint_encoder=[32, 64, 128],
        GNN_layer_num=9,
        use_clip=True,
    )
    model = model.cuda()
    model.load_state_dict(load_params(ckpt_path))
    model.eval()

    opt = {
        "root": path,
        "dstype": "contour",
        "split": "test",
        "is_png_seg": True,
        "color_redistribution_type": "gt",
    }
    dataset = PaintBucketSegMat(opt)
    dataloader = data.DataLoader(dataset, batch_size=1)

    model_inference = ModelInference(model, dataloader)
    save_path = path.replace("dataset/test", "results")
    model_inference.inference_frame_by_frame(save_path, accu=True, self_prop=True)
