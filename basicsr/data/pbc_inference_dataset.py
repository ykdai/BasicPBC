import numpy as np
import os
import os.path as osp
import torch
import torch.utils.data as data
from glob import glob

from basicsr.utils.registry import DATASET_REGISTRY
from paint.utils import read_img_2_np, read_seg_2_np, recolorize_seg, recolorize_gt


class AnimeInferenceDataset(data.Dataset):
    def __init__(self):
        self.data_list = []

    def _square_img_data(self, line, seg, gt=None, border=16):
        # Crop the content
        mask = np.any(line != [255, 255, 255], axis=-1)  # assume background is white
        coords = np.argwhere(mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        h, w = line.shape[:2]
        y_min, x_min = max(0, y_min - border), max(0, x_min - border)  # Extend border
        y_max, x_max = min(h, y_max + border), min(w, x_max + border)

        line = line[y_min : y_max + 1, x_min : x_max + 1]
        seg = seg[y_min : y_max + 1, x_min : x_max + 1]
        if gt is not None:
            gt = gt[y_min : y_max + 1, x_min : x_max + 1]

        # Pad to square
        nh, nw = line.shape[:2]
        diff = abs(nh - nw)
        pad1, pad2 = diff // 2, diff - diff // 2

        if nh > nw:
            # Width is smaller, pad left and right
            line = np.pad(line, ((0, 0), (pad1, pad2), (0, 0)), constant_values=255)
            seg = np.pad(seg, ((0, 0), (pad1, pad2)), constant_values=0)  # 0 will be ignored
            if gt is not None:
                gt = np.pad(gt, ((0, 0), (pad1, pad2), (0, 0)), mode="edge")
        else:
            # Height is smaller, pad top and bottom
            line = np.pad(line, ((pad1, pad2), (0, 0), (0, 0)), constant_values=255)
            seg = np.pad(seg, ((pad1, pad2), (0, 0)), constant_values=0)
            if gt is not None:
                gt = np.pad(gt, ((pad1, pad2), (0, 0), (0, 0)), mode="edge")

        return line, seg, gt if gt is not None else None

    def _process_seg(self, seg):
        seg_list = np.unique(seg[seg != 0])

        h, w = seg.shape
        hh = np.arange(h)
        ww = np.arange(w)
        xx, yy = np.meshgrid(ww, hh)

        keypoints = []
        centerpoints = []
        numpixels = []
        seg_relabeled = np.zeros_like(seg)

        for i, seg_idx in enumerate(seg_list):
            mask = seg == seg_idx

            xs = xx[mask]
            ys = yy[mask]
            xmin = xs.min()
            xmax = xs.max()
            ymin = ys.min()
            ymax = ys.max()
            xmean = xs.mean()
            ymean = ys.mean()
            keypoints.append([xmin, xmax, ymin, ymax])
            centerpoints.append([xmean, ymean])
            numpixels.append(mask.sum())

            seg_relabeled[mask] = i + 1  # 0 is for black line, start from 1

        keypoints = np.stack(keypoints)
        centerpoints = np.stack(centerpoints)
        numpixels = np.stack(numpixels)

        return keypoints, centerpoints, numpixels, seg_relabeled

    def __getitem__(self, index):

        index = index % len(self.data_list)

        file_name = self.data_list[index]["file_name"]
        file_name_ref = self.data_list[index]["file_name_ref"]

        # read images
        line = read_img_2_np(self.data_list[index]["line"])
        line_ref = read_img_2_np(self.data_list[index]["line_ref"])

        seg = read_seg_2_np(self.data_list[index]["seg"])
        seg_ref = read_seg_2_np(self.data_list[index]["seg_ref"])

        gt_ref = self.data_list[index]["gt_ref"]
        gt_ref = read_img_2_np(gt_ref) if gt_ref is not None else None

        line, seg, _ = self._square_img_data(line, seg)
        line_ref, seg_ref, gt_ref = self._square_img_data(line_ref, seg_ref, gt_ref)

        keypoints, centerpoints, numpixels, seg = self._process_seg(seg)
        keypoints_ref, centerpoints_ref, numpixels_ref, seg_ref = self._process_seg(seg_ref)

        # np to tensor
        line = torch.from_numpy(line).permute(2, 0, 1) / 255.0
        line_ref = torch.from_numpy(line_ref).permute(2, 0, 1) / 255.0
        seg = torch.from_numpy(seg)[None]
        seg_ref = torch.from_numpy(seg_ref)[None]

        recolorized_img = recolorize_seg(seg_ref) if gt_ref is None else recolorize_gt(gt_ref)

        return {
            "file_name": file_name,
            "file_name_ref": file_name_ref,
            "keypoints": keypoints,
            "keypoints_ref": keypoints_ref,
            "centerpoints": centerpoints,
            "centerpoints_ref": centerpoints_ref,
            "numpixels": numpixels,
            "numpixels_ref": numpixels_ref,
            "line": line,
            "line_ref": line_ref,
            "segment": seg,
            "segment_ref": seg_ref,
            "recolorized_img": recolorized_img,
        }

    def __rmul__(self, v):
        self.data_list = v * self.data_list
        return self

    def __len__(self):
        return len(self.data_list)


@DATASET_REGISTRY.register()
class PaintBucketInferenceDataset(AnimeInferenceDataset):
    def __init__(self, opt):
        super(PaintBucketInferenceDataset, self).__init__()

        self.opt = opt
        self.root = opt["root"]
        self.multi_clip = opt["multi_clip"] if "multi_clip" in opt else False

        if not self.multi_clip:
            character_paths = [self.root]
        else:
            character_paths = [osp.join(self.root, character) for character in os.listdir(self.root)]

        for character_path in character_paths:

            line_root = osp.join(character_path, "line")
            line_list = sorted(glob(osp.join(line_root, "*.png")))

            L = len(line_list)

            for i in range(L - 1):

                file_name, _ = osp.splitext(line_list[i + 1])
                line = line_list[i + 1]
                seg = line.replace("line", "seg")
                file_name_ref, _ = osp.splitext(line_list[i])
                line_ref = line_list[i]
                seg_ref = line_ref.replace("line", "seg")
                gt_ref = line_ref.replace("line", "gt")
                gt_ref = gt_ref if osp.exists(gt_ref) else None

                data_sample = {
                    "file_name": file_name,
                    "line": line,
                    "seg": seg,
                    "file_name_ref": file_name_ref,
                    "line_ref": line_ref,
                    "seg_ref": seg_ref,
                    "gt_ref": gt_ref,
                }
                self.data_list += [data_sample]

        print("Length of line frames is ", len(self.data_list))
