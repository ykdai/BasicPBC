# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import cv2
import numpy as np
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
from collections import Counter
from glob import glob
from PIL import Image
from skimage import io
from torchvision.transforms import ColorJitter

from basicsr.utils.registry import DATASET_REGISTRY
from paint.utils import read_img_2_np, read_seg_2_np, recolorize_gt, recolorize_seg


class SegMatAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, aug_half=False):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.aug_half = aug_half
        # Only augment the second frame, use original first frame
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """Photometric augmentation"""

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """Occlusion augmentation"""

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0 : y0 + dy, x0 : x0 + dx, :] = mean_color

        return img1, img2

    # def spatial_transform(self, img1, img2, seg1, seg2, adj_dict1, adj_dict2, matching):
    def spatial_transform(self, img1, img2, seg1, seg2):
        # augmentation should contain crop, shift, scaling
        # when croped: seg index need to be changed to [0..Nc], lost item mismatching
        # when scaled: seg index need to be rescaled. if larger, doesn't matter, if smaller, index may vanish. delete it and reindex as above
        # when shift: doesn't matter
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum((self.crop_size[0] + 8) / float(ht), (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale

        x = np.random.rand()
        if x < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)
        if self.aug_half:
            scale_x = np.clip(scale_x, 1.0, None)
            scale_y = np.clip(scale_y, 1.0, None)

        # rescale the images
        if np.random.rand() < self.spatial_aug_prob:

            if self.aug_half and x < self.stretch_prob:
                pass
            else:
                img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                seg1 = cv2.resize(seg1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)

            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            seg2 = cv2.resize(seg2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                seg1 = seg1[:, ::-1]
                seg2 = seg2[:, ::-1]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                seg1 = seg1[::-1, :]
                seg2 = seg2[::-1, :]

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        img2 = img2[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        seg1 = seg1[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        seg2 = seg2[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]

        seg1_id_hist = Counter(seg1.reshape(-1)).most_common()
        seg2_id_hist = Counter(seg2.reshape(-1)).most_common()

        seg1_id = np.array([ii for (ii, _) in seg1_id_hist])
        seg2_id = np.array([ii for (ii, _) in seg2_id_hist])

        seg1_new = seg1.copy()
        seg2_new = seg2.copy()
        for ii in range(len(seg1_id)):
            seg1_new[seg1 == seg1_id[ii]] = ii
        for ii in range(len(seg2_id)):
            seg2_new[seg2 == seg2_id[ii]] = ii

        return img1, img2, seg1_new, seg2_new

    def __call__(self, img1, img2, seg1, seg2):
        seg1 = seg1 - 1
        seg2 = seg2 - 1
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, seg1, seg2 = self.spatial_transform(img1, img2, seg1, seg2)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        seg1 = np.ascontiguousarray(seg1) + 1
        seg2 = np.ascontiguousarray(seg2) + 1

        return img1, img2, seg1, seg2


class AnimeSegMatDataset(data.Dataset):
    def __init__(self, aug_params=None):
        if aug_params is not None:
            self.augmentor = SegMatAugmentor(**aug_params)
        else:
            self.augmentor = None

        self.is_png_seg = False
        self.split = None
        self.color_redistribution_type = None  # Recolorize gt/seg using randomly selected colors, used for the optical flow module

        self.gt_list = []
        self.line_list = []
        self.seg_list = []

    def __getitem__(self, index):

        file_name = self.line_list[index][0][:-4]

        index = index % len(self.line_list)

        # read images
        line = read_img_2_np(self.line_list[index][0])
        line_ref = read_img_2_np(self.line_list[index][1])

        # load segmetns
        if self.is_png_seg:
            seg = read_seg_2_np(self.seg_list[index][0])
            seg_ref = read_seg_2_np(self.seg_list[index][1])
        else:
            # The image is in npy file.
            seg = np.load(self.seg_list[index][0]).astype(np.int64)
            seg_ref = np.load(self.seg_list[index][1]).astype(np.int64)

        # autmentation
        if self.augmentor is not None:
            line, line_ref, seg, seg_ref = self.augmentor(line, line_ref, seg, seg_ref)

        keypoints = []
        keypoints_ref = []
        centerpoints = []
        centerpoints_ref = []
        numpixels = []
        numpixels_ref = []

        h, w = seg.shape
        hh = np.arange(h)
        ww = np.arange(w)
        xx, yy = np.meshgrid(ww, hh)

        sys.stdout.flush()

        # TODO
        seg_list = sorted(Counter(seg.reshape(-1)))
        for ii in seg_list:
            if ii == 0:
                continue  # 0 means the black line
            xs = xx[seg == ii]
            ys = yy[seg == ii]

            xmin = xs.min()
            xmax = xs.max()
            ymin = ys.min()
            ymax = ys.max()
            xmean = xs.mean()
            ymean = ys.mean()

            centerpoints.append([xmean, ymean])
            numpixels.append((seg == ii).sum())
            keypoints.append([xmin, xmax, ymin, ymax])

        seg_ref_list = sorted(Counter(seg_ref.reshape(-1)))
        for ii in seg_ref_list:
            if ii == 0:
                continue  # 0 means black line
            xs = xx[seg_ref == ii]
            ys = yy[seg_ref == ii]

            xmin = xs.min()
            xmax = xs.max()
            ymin = ys.min()
            ymax = ys.max()
            xmean = xs.mean()
            ymean = ys.mean()

            centerpoints_ref.append([xmean, ymean])
            numpixels_ref.append((seg_ref == ii).sum())
            keypoints_ref.append([xmin, xmax, ymin, ymax])

        keypoints = np.stack(keypoints)
        keypoints_ref = np.stack(keypoints_ref)
        centerpoints = np.stack(centerpoints)
        centerpoints_ref = np.stack(centerpoints_ref)
        numpixels = np.stack(numpixels)
        numpixels_ref = np.stack(numpixels_ref)

        # image output [0, 1]
        line = torch.from_numpy(line).permute(2, 0, 1).float() / 255.0
        line_ref = torch.from_numpy(line_ref).permute(2, 0, 1).float() / 255.0
        seg = torch.from_numpy(seg)[None]
        seg_ref = torch.from_numpy(seg_ref)[None]
        numpixels = torch.from_numpy(numpixels)[None]
        numpixels_ref = torch.from_numpy(numpixels_ref)[None]

        if self.color_redistribution_type == "seg":
            recolorized_img = recolorize_seg(seg_ref)
        elif self.color_redistribution_type == "gt":
            gt_ref = read_img_2_np(self.gt_list[index][1])
            recolorized_img = recolorize_gt(gt_ref)
        else:
            recolorized_img = torch.Tensor(0)

        return {
            "keypoints": keypoints,
            "keypoints_ref": keypoints_ref,
            "centerpoints": centerpoints,
            "centerpoints_ref": centerpoints_ref,
            "line": line,
            "line_ref": line_ref,
            "numpixels": numpixels,
            "numpixels_ref": numpixels_ref,
            "segment": seg,
            "segment_ref": seg_ref,
            "recolorized_img": recolorized_img,
            "file_name": file_name,
        }

    def __rmul__(self, v):
        self.gt_list = v * self.gt_list
        self.line_list = v * self.line_list
        self.seg_list = v * self.seg_list
        return self

    def __len__(self):
        return len(self.line_list)


@DATASET_REGISTRY.register()
class PaintBucketSegMat(AnimeSegMatDataset):
    def __init__(self, opt):
        # This class is mainly for inference.
        aug_params = opt["aug_params"] if "aug_params" in opt else None
        super(PaintBucketSegMat, self).__init__(aug_params)

        self.opt = opt
        root = opt["root"]
        dstype = opt["dstype"]
        self.is_png_seg = opt["is_png_seg"] if "is_png_seg" in opt else False

        self.split = opt["split"] if "split" in opt else None
        if self.split == "test":
            self.augmentor = None

        self.color_redistribution_type = opt["color_redistribution_type"] if "color_redistribution_type" in opt else None
        assert self.color_redistribution_type in [None, "gt", "seg"]

        for character in os.listdir(root):
            line_root = osp.join(root, character, "line")
            if dstype == "Frame_Anime":
                assert False, "Please use the line art module."
            else:
                line_list = sorted(glob(osp.join(line_root, "*.png")))

            seg_root = osp.join(root, character, "seg")
            if self.is_png_seg:
                seg_list = sorted(glob(osp.join(seg_root, "*.png")))
            else:
                seg_list = sorted(glob(osp.join(seg_root, "*.npy")))

            gt_root = osp.join(root, character, "gt")
            gt_list = sorted(glob(osp.join(gt_root, "*.png")))

            assert len(line_list) == len(seg_list), 'line number should match the seg number'
            assert len(line_list) == len(gt_list), 'line number should match the gt number'

            L = len(line_list)
            for i in range(L - 1):
                self.line_list += [[line_list[i + 1], line_list[i]]]
                self.seg_list += [[seg_list[i + 1], seg_list[i]]]
                self.gt_list += [[gt_list[i + 1], gt_list[i]]]

        print("Length of Testing Sequence is ", len(self.line_list))
