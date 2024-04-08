import argparse
import os
import torch
import torch.utils.data as data
from glob import glob
from os import path as osp
from skimage import io, measure
from torchvision.utils import save_image

from basicsr.archs.basicpbc_arch import BasicPBC
from basicsr.data.pbc_inference_dataset import PaintBucketInferenceDataset
from basicsr.models.pbc_model import ModelInference
from paint.colorlabel import ColorLabel
from paint.lineart import LineArt, trappedball_fill
from paint.utils import dump_json, np_2_labelpng, read_seg_2_np, recolorize_seg


def generate_seg(path, seg_type="default", radius=4, save_color_seg=False, multi_clip=False):
    if not multi_clip:
        clip_list = [path]
    else:
        clip_list = [osp.join(path, clip) for clip in os.listdir(path)]

    for clip_path in clip_list:
        line_folder = osp.join(clip_path, "line")
        seg_folder = osp.join(clip_path, "seg")
        seg_color_folder = osp.join(clip_path, "seg_color")
        os.makedirs(seg_folder, exist_ok=True)
        if save_color_seg:
            os.makedirs(seg_color_folder, exist_ok=True)

        for line_path in sorted(glob(osp.join(line_folder, "*.png"))):
            name = osp.split(line_path)[-1][:-4]
            seg_path = osp.join(seg_folder, name + ".png")
            seg_color_path = osp.join(seg_color_folder, name + ".png")
            if seg_type == "default":
                lineart = LineArt(io.imread(line_path))
                lineart.label_color_line()
                seg_np = lineart.label_img
                seg = np_2_labelpng(seg_np)
                io.imsave(seg_path, seg, check_contrast=False)
                if save_color_seg:
                    color_seg = recolorize_seg(torch.from_numpy(seg_np)[None])
                    save_image(color_seg, seg_color_path)
            elif seg_type == "trappedball":
                trappedball_fill(line_path, seg_color_path, radius, contour=True)
                color_label = ColorLabel()
                if int(name) == 0:
                    color_label.extract_label_map(seg_color_path.replace("seg_color", "gt"), seg_path, line_path, extract_seg=True)
                else:
                    color_label.extract_label_map(seg_color_path, seg_path, line_path, extract_seg=True)
            print(f"{seg_path} created.")

        extract_color_dict(clip_path)


def extract_color_dict(clip_path):
    seg0_path = sorted(glob(osp.join(clip_path, "seg", "*.png")))[0]
    gt0_path = sorted(glob(osp.join(clip_path, "gt", "*.png")))[0]
    seg = read_seg_2_np(seg0_path)
    gt = io.imread(gt0_path)

    color_dict = {}
    props = measure.regionprops(seg)
    for i in range(1, seg.max() + 1):
        pos = props[i - 1].coords[0]
        index_color = gt[pos[0], pos[1], :]
        color_dict[str(i)] = index_color.tolist()

    dump_json(color_dict, seg0_path.replace(".png", ".json"))


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
    parser.add_argument("--path", type=str, default="dataset/test/your_clip")
    parser.add_argument("--seg_type", choices=["default", "trappedball"], default="default")
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--save_color_seg", action="store_true")
    parser.add_argument("--multi_clip", action="store_true")
    args = parser.parse_args()

    path = args.path
    seg_type = args.seg_type
    radius = args.radius
    save_color_seg = args.save_color_seg
    multi_clip = args.multi_clip

    generate_seg(path, seg_type, radius, save_color_seg, multi_clip)

    ckpt_path = "ckpt/basicpbc.pth"
    model = BasicPBC(
        ch_in=6,
        descriptor_dim=128,
        keypoint_encoder=[32, 64, 128],
        GNN_layer_num=9,
        use_clip=True,
        encoder_resolution=(640, 640),
        raft_resolution=(320, 320),
        clip_resolution=(320, 320),
    )
    model = model.cuda()
    model.load_state_dict(load_params(ckpt_path))
    model.eval()

    opt = {"root": path, "multi_clip": multi_clip}
    dataset = PaintBucketInferenceDataset(opt)
    dataloader = data.DataLoader(dataset, batch_size=1)

    model_inference = ModelInference(model, dataloader)
    save_path = path.replace("dataset/test", "results")
    model_inference.inference_frame_by_frame(save_path, accu=True, self_prop=True)
