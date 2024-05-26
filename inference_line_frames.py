import argparse
import os
import torch
import torch.utils.data as data
from glob import glob
from os import path as osp
from skimage import io, measure
from torchvision.utils import save_image
from torchvision.utils import save_image

from basicsr.archs.basicpbc_arch import BasicPBC
from basicsr.archs.basicpbc_light_arch import BasicPBC_light
from basicsr.data.pbc_inference_dataset import PaintBucketInferenceDataset
from basicsr.models.pbc_model import ModelInference
from paint.colorlabel import ColorLabel
from paint.lineart import LineArt, trappedball_fill
from paint.utils import dump_json, np_2_labelpng, process_gt, read_line_2_np, read_seg_2_np, recolorize_seg
from paint.colorlabel import ColorLabel
from paint.lineart import LineArt, trappedball_fill
from paint.utils import dump_json, np_2_labelpng, process_gt, read_line_2_np, read_seg_2_np, recolorize_seg, colorize_label_image
import shutil

def extract_seg_from_color(color_img_path, line_path, seg_save_path):
    color_label = ColorLabel()
    color_label.extract_label_map(color_img_path, seg_save_path, line_path, extract_seg=True)


def extract_seg_from_line(line_path, seg_save_path, save_color_seg=False, color_save_path=None):
    lineart = LineArt(read_line_2_np(line_path))
    lineart.label_color_line()
    seg_np = lineart.label_img
    seg = np_2_labelpng(seg_np)
    io.imsave(seg_save_path, seg, check_contrast=False)
    if save_color_seg:
        color_seg = recolorize_seg(torch.from_numpy(seg_np)[None])
        save_image(color_seg, color_save_path)


def extract_color_dict(gt_path, seg_path):
    gt = io.imread(gt_path)
    seg = read_seg_2_np(seg_path)
    gt = process_gt(gt, seg)
    color_dict = {}
    props = measure.regionprops(seg)
    for i in range(1, seg.max() + 1):
        pos = props[i - 1].coords[0]
        index_color = gt[pos[0], pos[1], :]
        color_dict[str(i)] = index_color.tolist()
    save_path = seg_path.replace(".png", ".json")
    dump_json(color_dict, save_path)


def generate_seg(path, seg_type="default", radius=4, save_color_seg=False, multi_clip=False):
    if seg_type == "trappedball":
        save_color_seg = True

    if not multi_clip:
        clip_list = [path]
    else:
        clip_list = [osp.join(path, clip) for clip in os.listdir(path)]

    for clip_path in clip_list:
        gt_folder = osp.join(clip_path, "gt")
        gt_backup_folder = osp.join(clip_path, "gt_backup")
        line_folder = osp.join(clip_path, "line")
        seg_folder = osp.join(clip_path, "seg")
        seg_color_folder = osp.join(clip_path, "seg_color")

        gt_names = [osp.splitext(gt)[0] for gt in os.listdir(gt_folder)]
        os.makedirs(seg_folder, exist_ok=True)
        if save_color_seg:
            os.makedirs(seg_color_folder, exist_ok=True)
        
        if os.path.exists(gt_backup_folder):
            # Restore the backup gt.
            shutil.rmtree(gt_folder)
            shutil.copytree(gt_backup_folder, gt_folder) 
        else:
            # Back up the gt.
            shutil.copytree(gt_folder, gt_backup_folder)

        for line_path in sorted(glob(osp.join(line_folder, "*.png"))):
            name = osp.split(line_path)[-1][:-4]
            seg_path = osp.join(seg_folder, name + ".png")
            seg_color_path = osp.join(seg_color_folder, name + ".png")

            if seg_type == "default":
                extract_seg_from_line(line_path, seg_path, save_color_seg, seg_color_path)
            elif seg_type == "trappedball":
                trappedball_fill(line_path, seg_color_path, radius, contour=True)
                extract_seg_from_color(seg_color_path, line_path, seg_path)

            if name in gt_names:
                gt_path = osp.join(gt_folder, name + ".png")
                extract_color_dict(gt_path, seg_path)
                colorize_label_image(seg_path, seg_path.replace(".png", ".json"), gt_path)

            print(f"{seg_path} created.")


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
    parser.add_argument("--path", type=str, default="dataset/test/laughing_girl", help="path to your anime clip folder or folder containing multiple clips.")
    parser.add_argument("--mode", choices=["forward", "nearest"], default="forward", help="")
    parser.add_argument("--seg_type", choices=["default", "trappedball"], default="default", help="choose `trappedball` if line art not closed.")
    parser.add_argument("--skip_seg", action="store_true", help="used when `seg` already exists.")
    parser.add_argument("--radius", type=int, default=4, help="used together with `--seg_type trappedball`. Increase the value if unclosed pixels' high.")
    parser.add_argument("--save_color_seg", action="store_true", help="add this arg to show colorized segment results. It's a must when `trappedball` chosen.")
    parser.add_argument("--use_light_model", action="store_true", help="add this to use light-weighted model on low memory GPU.")
    parser.add_argument("--multi_clip", action="store_true", help="used for multi-clip inference. Set `path` to a folder where each sub-folder is a single clip.")
    parser.add_argument("--keep_line", action="store_true", help="used for keeping the original line in the final output.")
    parser.add_argument("--raft_res", type=int, default=320, help="change the resolution for the optical flow estimation. If the performance is bad on your case, you can change this to 640 to have a try.")

    args = parser.parse_args()

    path = args.path
    mode = args.mode
    seg_type = args.seg_type
    skip_seg = args.skip_seg
    radius = args.radius
    save_color_seg = args.save_color_seg
    use_light_model = args.use_light_model
    multi_clip = args.multi_clip
    raft_resolution= args.raft_res
    keep_line= args.keep_line

    if not skip_seg:
        generate_seg(path, seg_type, radius, save_color_seg, multi_clip)

    if use_light_model:
        ckpt_path = "ckpt/basicpbc_light.pth"
        model = BasicPBC_light(
            descriptor_dim=128,
            keypoint_encoder=[32, 64, 128],
            GNN_layer_num=6,
            token_scale_list=[1, 3],
            token_crop_size=64,
            use_clip=True,
        )
    else:
        ckpt_path = "ckpt/basicpbc.pth"
        model = BasicPBC(
            ch_in=6,
            descriptor_dim=128,
            keypoint_encoder=[32, 64, 128],
            GNN_layer_num=9,
            use_clip=True,
            encoder_resolution=(640, 640),
            raft_resolution=(raft_resolution, raft_resolution),
            clip_resolution=(320, 320),
        )

    model = model.cuda()
    model.load_state_dict(load_params(ckpt_path))
    model.eval()

    opt = {"root": path, "multi_clip": multi_clip, "mode": mode}
    dataset = PaintBucketInferenceDataset(opt)
    dataloader = data.DataLoader(dataset, batch_size=1)

    model_inference = ModelInference(model, dataloader)
    model_inference.inference_multi_gt(path,keep_line)
