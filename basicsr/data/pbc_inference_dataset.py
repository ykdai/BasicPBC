import numpy as np
import os
import os.path as osp
import torch
import torch.utils.data as data
from glob import glob

from basicsr.utils.registry import DATASET_REGISTRY
from paint.utils import read_img_2_np, read_seg_2_np, recolorize_seg


class AnimeInferenceDataset(data.Dataset):
    def __init__(self):
        self.data_list = []

    def _square_line_seg(self, line, seg, border=32):
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

        # Pad to square
        nh, nw = line.shape[:2]
        diff = abs(nh - nw)
        pad1, pad2 = diff // 2, diff - diff // 2

        if nh > nw:
            # Width is smaller, pad left and right
            line = np.pad(line, ((0, 0), (pad1, pad2), (0, 0)), constant_values=255)
            seg = np.pad(seg, ((0, 0), (pad1, pad2)), constant_values=1)  # 1 means background
        else:
            # Height is smaller, pad top and bottom
            line = np.pad(line, ((pad1, pad2), (0, 0), (0, 0)), constant_values=255)
            seg = np.pad(seg, ((pad1, pad2), (0, 0)), constant_values=1)  # 1 means background

        return line, seg

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

        height, width = line.shape[:2]
        line, seg = self._square_line_seg(line, seg)
        line_ref, seg_ref = self._square_line_seg(line_ref, seg_ref)

        keypoints, centerpoints, numpixels, seg = self._process_seg(seg)
        keypoints_ref, centerpoints_ref, numpixels_ref, seg_ref = self._process_seg(seg_ref)

        # np to tensor
        line = torch.from_numpy(line).permute(2, 0, 1) / 255.0
        line_ref = torch.from_numpy(line_ref).permute(2, 0, 1) / 255.0
        seg = torch.from_numpy(seg)[None]
        seg_ref = torch.from_numpy(seg_ref)[None]

        colored_seg_ref = recolorize_seg(seg_ref)

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
            "recolorized_img": colored_seg_ref,
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
            seg_root = osp.join(character_path, "seg")

            line_list = sorted(glob(osp.join(line_root, "*.png")))
            seg_list = sorted(glob(osp.join(seg_root, "*.png")))

            L = len(line_list)

            for i in range(L - 1):
                data_sample = {
                    "file_name": line_list[i + 1][:-4],
                    "line": line_list[i + 1],
                    "seg": seg_list[i + 1],
                    "file_name_ref": line_list[i][:-4],
                    "line_ref": line_list[i],
                    "seg_ref": seg_list[i],
                }
                self.data_list += [data_sample]

        # TODO
        print("Length of data sample list is", len(self.data_list))
