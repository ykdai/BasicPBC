# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import argparse
import cv2
import math
import networkx as nx
import numpy as np
import os
import os.path as osp
import random
import sys
import torch
import torch.utils.data as data

# import cv2
from collections import Counter
from glob import glob
from PIL import Image
from scipy.ndimage import zoom
from skimage import io
from torchvision.transforms import ColorJitter
from tqdm import tqdm
import json

from basicsr.data.transforms import Augmentor, AugmentorNonPair
from basicsr.utils.registry import DATASET_REGISTRY
from paint.colorbook import ColorBook
from paint.utils import (
    default_colorbook,
    generate_random_colors,
    labelpng_2_np,
    load_json,
    read_img_2_np,
    read_seg_2_np,
    recolorize_gt,
    recolorize_seg,
    process_line_anno,
)


def read_json(json_path):
    # Load the adjacency json and return the adjacency dict
    with open(json_path, 'r') as json_file:
        graph_data = json.load(json_file)
    return graph_data


class AnimeLabelSegDataset(data.Dataset):
    def __init__(self, aug_params=None):
        if aug_params is not None:
            print("Data augmentation loaded!")
            self.augmentor = Augmentor(**aug_params)
        else:
            self.augmentor = None

        self.pixel_threshold = None

        self.init_seed = False
        self.load_adj = False  # Load adjacency relationship
        self.shuffle_label = False  # Shuffle the index of the labels
        self.merge_label_prob = 0.0  # Merge the labels as the data augmentation
        self.use_colorbook = False
        self.use_short_token = False

        # image, seg, matching index etc
        self.image_list = []
        self.label_list = []
        self.seg_list = []
        self.color_list = []
        self.idx_list = []
        self.adj_list = []
        self.colorbook_list = []

    def __getitem__(self, index):

        file_name = self.image_list[index][0][:-4]

        index = index % len(self.image_list)
        valid = None

        # read images
        img1 = Image.open(self.image_list[index][0]).convert("RGB")
        img2 = Image.open(self.image_list[index][1]).convert("RGB")

        # load seg for target and label for reference
        # The label starts from -1 (background), 0 means black line
        label1 = labelpng_2_np(io.imread(self.seg_list[index][0]))
        label2 = labelpng_2_np(io.imread(self.label_list[index][1]))

        # target seg to label
        idx_map1 = read_json(self.idx_list[index][0])
        idx_map2 = read_json(self.idx_list[index][1])

        if self.use_colorbook:
            # ref seg to color
            colorbook = ColorBook(self.colorbook_list[index])
            color_dict1 = read_json(self.color_list[index][0])
            color_dict2 = read_json(self.color_list[index][1])

        if self.load_adj:
            adj_dict1 = read_json(self.adj_list[index][0])
            adj_dict2 = read_json(self.adj_list[index][1])
        else:
            adj_dict1 = None
            adj_dict2 = None

        # flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        label1 = np.array(label1).astype(np.int64)
        label2 = np.array(label2).astype(np.int64)

        # image --> 3 channels
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        # image output [-1, 1]
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
        label1 = torch.from_numpy(label1)[None]
        label2 = torch.from_numpy(label2)[None]

        # autmentation
        if self.augmentor is not None:
            img1, img2, label1, label2 = self.augmentor(img1, img2, label1, label2)
        label1 = label1.squeeze().numpy().astype(np.int64)
        label2 = label2.squeeze().numpy().astype(np.int64)

        # match seg1 to label2

        label1_list = Counter(label1.reshape(-1))  # We remove the sorted to keep the mapping in order
        label2_list = Counter(label2.reshape(-1))

        if self.pixel_threshold:
            label1_list = {k: v for k, v in label1_list.items() if v >= self.pixel_threshold}
            label2_list = {k: v for k, v in label2_list.items() if v >= self.pixel_threshold}

        label1_list = [x for x in label1_list if x != 0]  # 0 means black line in our training set
        label2_list = [x for x in label2_list if x != 0]

        if self.shuffle_label:
            label2_list = sorted(label2_list)
            background_index, label2_list_sliced = label2_list[0], label2_list[1:]
            random.shuffle(label1_list)
            random.shuffle(label2_list_sliced)
            label2_list = [background_index, *label2_list_sliced]

        # [-1,2,4,5,6] & [-1,2,3,5,7] -> {-1:-1, 2:2, 4:0, 5:5, 6:0}
        # mat_dict = {x: x if x in label2_list else 0 for x in label1_list}
        # mat_dict_b = {x: x if x in label1_list else 0 for x in label2_list} # backward

        # # [-1,2,4,5,6] & [-1,2,3,5,7] -> [0, 1, -1, 3, -1], -1 means not match
        # mat_index = [label2_list.index(item) if item in label2_list else -1 for item in label1_list]

        # match seg1 to label2

        # segment idx to label idx
        def idx(x):
            return idx_map1[str(x)][1]

        def color1(x):
            return color_dict1[str(x)]

        def color2(x):
            # label idx to seg idx
            for seg_idx, (_, label_idx) in idx_map2.items():
                if x == label_idx:
                    return color_dict2[str(seg_idx)]
            return [0, 0, 0, 0]

        mat_index = [label2_list.index(idx(x)) if idx(x) in label2_list else -1 for x in label1_list]
        mat_index = np.array(mat_index).astype(np.int64)

        if self.load_adj:
            adj2 = {}
            for i, ii in enumerate(label2_list):
                adj2[i] = [label2_list.index(item) if item in label2_list else -1 for item in adj_dict2[str(ii)]]
                adj2[i] = [x for x in adj2[i] if x != -1]

        kpt1 = []
        kpt2 = []
        cpt1 = []
        cpt2 = []
        numpt1 = []
        numpt2 = []
        color_names1 = []
        color_names2 = []

        h, w = label1.shape[-2:]
        hh = np.arange(h)
        ww = np.arange(w)
        xx, yy = np.meshgrid(ww, hh)

        # sys.stdout.flush()

        label1_relabeled = np.zeros_like(label1)
        label2_relabeled = np.zeros_like(label2)

        for i, ii in enumerate(label1_list):
            mask = label1 == ii
            xs = xx[mask]
            ys = yy[mask]

            xmin = xs.min()
            xmax = xs.max()
            ymin = ys.min()
            ymax = ys.max()
            xmean = xs.mean()
            ymean = ys.mean()

            cpt1.append([xmean, ymean])
            numpt1.append(mask.sum())
            kpt1.append([xmin, xmax, ymin, ymax])
            label1_relabeled[mask] = i + 1  # 0 is for the black line,starts from 1
            if self.use_colorbook:
                color_names1.append(colorbook.get_color_name(color1(ii)))

        label2_index = 1
        merged_list = []  # like [0,1,2,2,3,1], index -> merged label's index, 0 means background

        for i, ii in enumerate(label2_list):
            # label2_list starts from 1
            mask = label2 == ii
            merge_flag = np.random.choice([0, 1], p=[1 - self.merge_label_prob, self.merge_label_prob])
            if self.merge_label_prob > 0 and merge_flag and i > 5:
                # randomly merge the label with the previous one
                if self.load_adj:
                    # just merge the adjacent labels, find union of adj2[i] and [1, label2_index] (0 means background)
                    filtered_labels = [x for x in adj2[i] if 0 < x and x < i]
                    if len(filtered_labels) == 0:
                        merge_flag = 0
                    else:
                        selected_label = merged_list[np.random.choice(filtered_labels)] + 1  # merged list 0 is background
                else:
                    selected_label = np.random.randint(2, label2_index)  # 0 means black line, and 1 means background
                if merge_flag:
                    label2_relabeled[mask] = selected_label
                    mask = label2_relabeled == selected_label

            xs = xx[mask]
            ys = yy[mask]

            xmin = xs.min()
            xmax = xs.max()
            ymin = ys.min()
            ymax = ys.max()
            xmean = xs.mean()
            ymean = ys.mean()

            if self.merge_label_prob > 0 and merge_flag and i > 5:
                cpt2[selected_label - 1] = [xmean, ymean]
                numpt2[selected_label - 1] = mask.sum()
                kpt2[selected_label - 1] = [xmin, xmax, ymin, ymax]
                if self.use_colorbook:
                    color_names2[selected_label - 1] = colorbook.get_color_name(color2(ii))

            else:
                cpt2.append([xmean, ymean])
                numpt2.append(mask.sum())
                kpt2.append([xmin, xmax, ymin, ymax])
                label2_relabeled[mask] = label2_index  # 0 is for the black line
                selected_label = label2_index
                label2_index += 1
                if self.use_colorbook:
                    color_names2.append(colorbook.get_color_name(color2(ii)))

            # All the token which marks
            mat_index = np.where(mat_index == i, selected_label - 1, mat_index)
            merged_list.append(selected_label - 1)

        label1 = label1_relabeled
        label2 = label2_relabeled

        kpt1 = np.stack(kpt1)
        kpt2 = np.stack(kpt2)
        cpt1 = np.stack(cpt1)
        cpt2 = np.stack(cpt2)
        numpt1 = np.stack(numpt1)
        numpt2 = np.stack(numpt2)

        label1 = torch.from_numpy(label1)[None]
        label2 = torch.from_numpy(label2)[None]
        numpt1 = torch.from_numpy(numpt1)[None]
        numpt2 = torch.from_numpy(numpt2)[None]

        mat_index = torch.from_numpy(mat_index).float()

        if self.use_colorbook and self.use_short_token:
            color_names1 = [t.split()[-1] for t in color_names1]
            color_names2 = [t.split()[-1] for t in color_names2]

        return {
            "keypoints0": kpt1,
            "keypoints1": kpt2,
            "center_points0": cpt1,
            "center_points1": cpt2,
            "image0": img1,
            "image1": img2,
            "num0": numpt1,
            "num1": numpt2,
            "segment0": label1,
            "segment1": label2,
            "text0": color_names1,
            "text1": color_names2,
            "all_matches": mat_index,
            "file_name": file_name,
        }

    def __rmul__(self, v):
        self.index_list = v * self.index_list
        self.label_list = v * self.label_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


@DATASET_REGISTRY.register()
class PaintBucketSeqRefLabelSegDataset(AnimeLabelSegDataset):
    def __init__(self, opt):
        # This class is mainly for inference.
        aug_params = opt["aug_params"] if "aug_params" in opt else None
        super(PaintBucketSeqRefLabelSegDataset, self).__init__(aug_params)

        self.opt = opt
        self.root = opt["root"]
        self.is_test = opt["is_test"] if "is_test" in opt else False
        self.load_adj = opt["load_adj"] if "load_adj" in opt else False
        self.shuffle_label = opt["shuffle_label"] if "shuffle_label" in opt else False
        self.merge_label_prob = opt["merge_label_prob"] if "merge_label_prob" in opt else 0.0
        self.pixel_threshold = opt["pixel_threshold"] if "pixel_threshold" in opt else None
        self.index_list = None
        self.use_colorbook = opt["use_colorbook"] if "use_colorbook" in opt else False
        self.use_short_token = opt["use_short_token"] if "use_short_token" in opt else False
        self.truncate = opt["truncate"] if "truncate" in opt else None

        for scene in tqdm(os.listdir(self.root)):
            # if is_test:
            #     label_root = osp.join(root, scene, 'seg')
            # else:
            #     label_root = osp.join(root, scene, 'label')
            line_root = osp.join(self.root, scene, "line_black")
            label_root = osp.join(self.root, scene, "label")
            seg_root = osp.join(self.root, scene, "seg")
            idx_root = osp.join(self.root, scene, "json_index")

            image_list = sorted(glob(osp.join(line_root, "*.png")))
            label_list = sorted(glob(osp.join(label_root, "*.png")))
            seg_list = sorted(glob(osp.join(seg_root, "*.png")))
            idx_list = sorted(glob(osp.join(idx_root, "*.json")))

            if self.use_colorbook:
                colorbook_root = osp.join(self.root, scene, "colorbook.yml")
                color_root = osp.join(self.root, scene, "json_color")
                color_list = sorted(glob(osp.join(color_root, "*.json")))

            if self.load_adj:
                adj_root = osp.join(self.root, scene, "adj")
                adj_list = sorted(glob(osp.join(adj_root, "*.json")))

            seq_len = len(image_list)

            # Change the test seq to estimate the next frame
            self.sep_list = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 150, 200, 250, 400, 512]
            for sep in self.sep_list:
                for i in range(seq_len - sep):
                    if self.truncate and i >= self.truncate:
                        break
                    if self.load_adj:
                        self.adj_list += [[adj_list[i + sep], adj_list[i]]]
                    self.image_list += [[image_list[i + sep], image_list[i]]]
                    self.label_list += [[label_list[i + sep], label_list[i]]]
                    self.seg_list += [[seg_list[i + sep], seg_list[i]]]
                    self.idx_list += [[idx_list[i + sep], idx_list[i]]]
                    if self.use_colorbook:
                        self.color_list += [[color_list[i + sep], color_list[i]]]
                        self.colorbook_list += [colorbook_root]

            """
            for j in range(seq_len):
                for i in range(seq_len):
                    if self.load_adj:
                        self.adj_list += [ [adj_list[j], adj_list[i]] ]
                    self.image_list += [ [image_list[j], image_list[i]] ]
                    self.label_list += [ [label_list[j], label_list[i]] ]
                    self.seg_list += [ [seg_list[j], seg_list[i]] ]
                    self.idx_list += [ [idx_list[j], idx_list[i]] ]
                    if self.use_colorbook:
                        self.color_list += [ [color_list[j], color_list[i]] ]
                        self.colorbook_list += [colorbook_root]
            """
        print("Len of Seq is", len(self.image_list))


class AnimeLabelSegDatasetNonPair(data.Dataset):
    def __init__(self, aug_params=None):
        if aug_params is not None:
            print("Data augmentation loaded!")
            self.augmentor = AugmentorNonPair(**aug_params)
        else:
            self.augmentor = None

        self.pixel_threshold = None

        self.init_seed = False
        self.load_adj = False  # Load adjacency relationship
        self.shuffle_label = False  # Shuffle the index of the labels
        # self.merge_label_prob = 0.0 # Merge the labels as the data augmentation
        self.use_colorbook = False
        self.use_short_token = False

        # image, seg, matching index etc
        self.image_list = []
        self.label_list = []
        self.seg_list = []
        self.color_list = []
        self.idx_list = []
        self.adj_list = []
        self.colorbook_list = []

    def __getitem__(self, index):

        file_name = self.image_list[index][:-4]

        index = index % len(self.image_list)
        valid = None

        # read images
        img = Image.open(self.image_list[index]).convert("RGB")

        # load seg for target and label for reference
        # The label starts from -1 (background), 0 means black line
        label = labelpng_2_np(io.imread(self.seg_list[index]))

        # target seg to label
        idx_map = read_json(self.idx_list[index])

        if self.use_colorbook:
            # ref seg to color
            colorbook = ColorBook(self.colorbook_list[index])
            color_dict = read_json(self.color_list[index])

        if self.load_adj:
            adj_dict = read_json(self.adj_list[index])
        else:
            adj_dict = None

        # flow = np.array(flow).astype(np.float32)
        img = np.array(img).astype(np.uint8)
        label = np.array(label).astype(np.int64)

        # image --> 3 channels
        if len(img.shape) == 2:
            img = np.tile(img[..., None], (1, 1, 3))
        else:
            img = img[..., :3]

        # image output [-1, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(label)[None]

        # autmentation
        if self.augmentor is not None:
            img, label = self.augmentor(img, label)
        label = label.squeeze().numpy().astype(np.int64)

        # match seg1 to label2

        label_list = Counter(label.reshape(-1))  # We remove the sorted to keep the mapping in order

        if self.pixel_threshold:
            label_list = {k: v for k, v in label_list.items() if v >= self.pixel_threshold}

        label_list = [x for x in label_list if x != 0]  # 0 means black line in our training set

        if self.shuffle_label:
            random.shuffle(label_list)

        # [-1,2,4,5,6] & [-1,2,3,5,7] -> {-1:-1, 2:2, 4:0, 5:5, 6:0}
        # mat_dict = {x: x if x in label2_list else 0 for x in label1_list}
        # mat_dict_b = {x: x if x in label1_list else 0 for x in label2_list} # backward

        # # [-1,2,4,5,6] & [-1,2,3,5,7] -> [0, 1, -1, 3, -1], -1 means not match
        # mat_index = [label2_list.index(item) if item in label2_list else -1 for item in label1_list]

        # match seg1 to label2

        # segment idx to label idx
        def idx(x):
            return idx_map[str(x)][1]

        def color(x):
            return color_dict[str(x)]

        # mat_index = [label2_list.index(idx(x)) if idx(x) in label2_list else -1 for x in label1_list]
        # mat_index = np.array(mat_index).astype(np.int64)

        # if self.load_adj:
        #     adj2={}
        #     for i,ii in enumerate(label2_list):
        #         adj2[i]=[label2_list.index(item) if item in label2_list else -1 for item in adj_dict2[str(ii)]]
        #         adj2[i] = [x for x in adj2[i] if x != -1]

        kpt = []
        cpt = []
        numpt = []
        color_names = []

        h, w = label.shape[-2:]
        hh = np.arange(h)
        ww = np.arange(w)
        xx, yy = np.meshgrid(ww, hh)

        # sys.stdout.flush()

        label_relabeled = np.zeros_like(label)

        for i, ii in enumerate(label_list):
            mask = label == ii
            xs = xx[mask]
            ys = yy[mask]

            xmin = xs.min()
            xmax = xs.max()
            ymin = ys.min()
            ymax = ys.max()
            xmean = xs.mean()
            ymean = ys.mean()

            cpt.append([xmean, ymean])
            numpt.append(mask.sum())
            kpt.append([xmin, xmax, ymin, ymax])
            label_relabeled[mask] = i + 1  # 0 is for the black line,starts from 1
            if self.use_colorbook:
                color_names.append(colorbook.get_color_name(color(ii)))

        label = label_relabeled

        kpt = np.stack(kpt)
        cpt = np.stack(cpt)
        numpt = np.stack(numpt)

        label = torch.from_numpy(label)[None]
        numpt = torch.from_numpy(numpt)[None]

        # mat_index = torch.from_numpy(mat_index).float()

        if self.use_colorbook and self.use_short_token:
            color_names = [t.split()[-1] for t in color_names]

        return {
            "keypoints": kpt,
            "center_points": cpt,
            "image": img,
            "num": numpt,
            "segment": label,
            "text": color_names,
            "file_name": file_name,
        }

    def __rmul__(self, v):
        self.index_list = v * self.index_list
        self.label_list = v * self.label_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


@DATASET_REGISTRY.register()
class PaintBucketRefLabelSegDatasetNonPair(AnimeLabelSegDatasetNonPair):
    def __init__(self, opt):
        # This class is mainly for inference.
        aug_params = opt["aug_params"] if "aug_params" in opt else None
        super(PaintBucketRefLabelSegDatasetNonPair, self).__init__(aug_params)

        self.opt = opt
        self.root = opt["root"]
        self.is_test = opt["is_test"] if "is_test" in opt else False
        self.load_adj = opt["load_adj"] if "load_adj" in opt else False
        self.shuffle_label = opt["shuffle_label"] if "shuffle_label" in opt else False
        # self.merge_label_prob=opt['merge_label_prob'] if 'merge_label_prob' in opt else 0.0
        self.pixel_threshold = opt["pixel_threshold"] if "pixel_threshold" in opt else None
        self.index_list = None
        self.use_colorbook = opt["use_colorbook"] if "use_colorbook" in opt else False
        self.use_short_token = opt["use_short_token"] if "use_short_token" in opt else False

        for char in os.listdir(self.root):
            # if is_test:
            #     label_root = osp.join(root, scene, 'seg')
            # else:
            #     label_root = osp.join(root, scene, 'label')
            line_root = osp.join(self.root, char, "line_black")
            label_root = osp.join(self.root, char, "label")
            seg_root = osp.join(self.root, char, "seg")
            idx_root = osp.join(self.root, char, "json_index")

            line_ref_root = osp.join(self.root, char, "ref", "line_black")
            label_ref_root = osp.join(self.root, char, "ref", "label")
            seg_ref_root = osp.join(self.root, char, "ref", "seg")
            idx_ref_root = osp.join(self.root, char, "ref", "json_index")

            image_list = sorted(glob(osp.join(line_root, "*.png")))
            label_list = sorted(glob(osp.join(label_root, "*.png")))
            seg_list = sorted(glob(osp.join(seg_root, "*.png")))
            idx_list = sorted(glob(osp.join(idx_root, "*.json")))

            image_ref_list = sorted(glob(osp.join(line_ref_root, "*.png")))
            label_ref_list = sorted(glob(osp.join(label_ref_root, "*.png")))
            seg_ref_list = sorted(glob(osp.join(seg_ref_root, "*.png")))
            idx_ref_list = sorted(glob(osp.join(idx_ref_root, "*.json")))

            if self.use_colorbook:
                colorbook_root = osp.join(self.root, char, "colorbook.yml")
                color_root = osp.join(self.root, char, "json_color")
                color_ref_root = osp.join(self.root, char, "ref", "json_color")
                color_list = sorted(glob(osp.join(color_root, "*.json")))
                color_ref_list = sorted(glob(osp.join(color_ref_root, "*.json")))

            if self.load_adj:
                adj_root = osp.join(self.root, char, "adj")
                adj_ref_root = osp.join(self.root, char, "ref", "adj")
                adj_list = sorted(glob(osp.join(adj_root, "*.json")))
                adj_ref_list = sorted(glob(osp.join(adj_ref_root, "*.json")))

            seq_len = len(image_list)

            # Change the test seq to estimate the next frame
            for i in range(seq_len):
                if self.load_adj:
                    self.adj_list += [adj_list[i]]
                self.image_list += [image_list[i]]
                self.label_list += [label_list[i]]
                self.seg_list += [seg_list[i]]
                self.idx_list += [idx_list[i]]
                if self.use_colorbook:
                    self.color_list += [color_list[i]]
                    self.colorbook_list += [colorbook_root]

        print("Len of Seq is", len(self.image_list))


@DATASET_REGISTRY.register()
class PaintBucketRefLabelSegDataset(AnimeLabelSegDataset):
    def __init__(self, opt):
        # This class is mainly for inference.
        aug_params = opt["aug_params"] if "aug_params" in opt else None
        super(PaintBucketRefLabelSegDataset, self).__init__(aug_params)

        self.opt = opt
        self.root = opt["root"]
        self.is_test = opt["is_test"] if "is_test" in opt else False
        self.load_adj = opt["load_adj"] if "load_adj" in opt else False
        self.shuffle_label = opt["shuffle_label"] if "shuffle_label" in opt else False
        self.merge_label_prob = opt["merge_label_prob"] if "merge_label_prob" in opt else 0.0
        self.pixel_threshold = opt["pixel_threshold"] if "pixel_threshold" in opt else None
        self.index_list = None
        self.use_colorbook = opt["use_colorbook"] if "use_colorbook" in opt else False
        self.use_short_token = opt["use_short_token"] if "use_short_token" in opt else False

        for char in os.listdir(self.root):
            # if is_test:
            #     label_root = osp.join(root, scene, 'seg')
            # else:
            #     label_root = osp.join(root, scene, 'label')
            line_root = osp.join(self.root, char, "line_black")
            label_root = osp.join(self.root, char, "label")
            seg_root = osp.join(self.root, char, "seg")
            idx_root = osp.join(self.root, char, "json_index")

            line_ref_root = osp.join(self.root, char, "ref", "line_black")
            label_ref_root = osp.join(self.root, char, "ref", "label")
            seg_ref_root = osp.join(self.root, char, "ref", "seg")
            idx_ref_root = osp.join(self.root, char, "ref", "json_index")

            image_list = sorted(glob(osp.join(line_root, "*.png")))
            label_list = sorted(glob(osp.join(label_root, "*.png")))
            seg_list = sorted(glob(osp.join(seg_root, "*.png")))
            idx_list = sorted(glob(osp.join(idx_root, "*.json")))

            image_ref_list = sorted(glob(osp.join(line_ref_root, "*.png")))
            label_ref_list = sorted(glob(osp.join(label_ref_root, "*.png")))
            seg_ref_list = sorted(glob(osp.join(seg_ref_root, "*.png")))
            idx_ref_list = sorted(glob(osp.join(idx_ref_root, "*.json")))

            if self.use_colorbook:
                colorbook_root = osp.join(self.root, char, "colorbook.yml")
                color_root = osp.join(self.root, char, "json_color")
                color_ref_root = osp.join(self.root, char, "ref", "json_color")
                color_list = sorted(glob(osp.join(color_root, "*.json")))
                color_ref_list = sorted(glob(osp.join(color_ref_root, "*.json")))

            if self.load_adj:
                adj_root = osp.join(self.root, char, "adj")
                adj_ref_root = osp.join(self.root, char, "ref", "adj")
                adj_list = sorted(glob(osp.join(adj_root, "*.json")))
                adj_ref_list = sorted(glob(osp.join(adj_ref_root, "*.json")))

            seq_len = len(image_list)

            # Change the test seq to estimate the next frame
            for i in range(seq_len):
                if self.load_adj:
                    self.adj_list += [[adj_list[i], adj_ref_list[0]]]
                self.image_list += [[image_list[i], image_ref_list[0]]]
                self.label_list += [[label_list[i], label_ref_list[0]]]
                self.seg_list += [[seg_list[i], seg_ref_list[0]]]
                self.idx_list += [[idx_list[i], idx_ref_list[0]]]
                if self.use_colorbook:
                    self.color_list += [[color_list[i], color_ref_list[0]]]
                    self.colorbook_list += [colorbook_root]

        print("Len of Seq is", len(self.image_list))


### NEW VERSION ###
class LineSegAugmentor:
    def __init__(self, rotate, resize, crop, rotate_prob):
        if resize < crop:
            raise ValueError("Crop size must be smaller than resize size.")
        self.rotate_degree = rotate
        self.resize_range = [crop, resize]
        self.crop_size = crop
        self.rotate_prob = rotate_prob

    def rotate_and_scale(self, image, angle):
        rows, cols = image.shape[:2]
        center = (cols / 2, rows / 2)
        angle_rad = np.deg2rad(angle)
        scale_factor = abs(np.sin(angle_rad)) + abs(np.cos(angle_rad))

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)

        transformed_image = cv2.warpAffine(
            image,
            rotation_matrix,
            (cols, rows),
            borderMode=cv2.BORDER_CONSTANT,
            flags=cv2.INTER_NEAREST,
        )
        return transformed_image

    def resize(self, image, resize_size):
        resized_image = cv2.resize(image, (resize_size, resize_size), interpolation=cv2.INTER_NEAREST)
        return resized_image

    def crop(self, image, start_x, start_y):
        return image[start_y : start_y + self.crop_size, start_x : start_x + self.crop_size]

    def __call__(self, line, seg, extra=None):

        apply_rotation = np.random.rand() <= self.rotate_prob
        angle = np.random.uniform(-self.rotate_degree, self.rotate_degree) if apply_rotation else 0
        resize_size = np.random.randint(self.resize_range[0], self.resize_range[1] + 1)
        start_x = np.random.randint(0, resize_size - self.crop_size + 1)
        start_y = np.random.randint(0, resize_size - self.crop_size + 1)

        line_transformed = self.rotate_and_scale(line, angle)
        seg_transformed = self.rotate_and_scale(seg, angle)

        line_resized = self.resize(line_transformed, resize_size)
        seg_resized = self.resize(seg_transformed, resize_size)

        line_cropped = self.crop(line_resized, start_x, start_y)
        seg_cropped = self.crop(seg_resized, start_x, start_y)

        if extra is not None:
            extra_transformed = self.rotate_and_scale(extra, angle)
            extra_resized = self.resize(extra_transformed, resize_size)
            extra_cropped = self.crop(extra_resized, start_x, start_y)

        # in case cropped frame excludes prospect character
        while len(np.unique(seg_cropped)) < 3:
            start_x = np.random.randint(0, resize_size - self.crop_size + 1)
            start_y = np.random.randint(0, resize_size - self.crop_size + 1)
            line_cropped = self.crop(line_resized, start_x, start_y)
            seg_cropped = self.crop(seg_resized, start_x, start_y)
            if extra is not None:
                extra_cropped = self.crop(extra_resized, start_x, start_y)

        return line_cropped, seg_cropped, extra_cropped if extra is not None else None


class AnimeTagSegDataset(data.Dataset):
    def __init__(self, aug_params=None):
        if aug_params is not None:
            self.augmentor = LineSegAugmentor(**aug_params)
        else:
            self.augmentor = None

        self.split = None
        self.shuffle_label = False  # Shuffle the index of the labels

        self.data_list = []

    def __getitem__(self, index):

        index = index % len(self.data_list)

        file_name = self.data_list[index]["file_name"]
        file_name_refs = self.data_list[index]["file_name_refs"]
        colorbook = ColorBook(self.data_list[index]["colorbook"])
        ref_length = self.data_list[index]["ref_length"]

        # read images
        # gt = read_img_2_np(self.data_list[index]["gt"])
        # gt_refs = [read_img_2_np(gt_ref) for gt_ref in self.data_list[index]["gt_refs"]]

        line = read_img_2_np(self.data_list[index]["line"])
        line_refs = [read_img_2_np(line_ref) for line_ref in self.data_list[index]["line_refs"]]
        gt_refs = [read_img_2_np(gt_ref) for gt_ref in self.data_list[index]["gt_refs"]]

        seg = read_seg_2_np(self.data_list[index]["seg"])
        seg_refs = [read_seg_2_np(seg_ref) for seg_ref in self.data_list[index]["seg_refs"]]

        if self.hint_name is not None:
            hint = read_img_2_np(self.data_list[index]["hint"])
            hint = process_line_anno(hint,seg,use_color=False) # [H,W] , 0 is background
        else:
            hint = np.zeros_like(line[:,:,0])

        color_map = load_json(self.data_list[index]["color"])
        color_map = {int(idx): color for idx, color in color_map.items()}
        color_map_refs = [load_json(color_ref) for color_ref in self.data_list[index]["color_refs"]]
        color_map_refs = [{int(idx): color for idx, color in color_map_ref.items()} for color_map_ref in color_map_refs]

        if self.data_list[index]["index"] is not None:
            seg2label = load_json(self.data_list[index]["index"])
            seg2label = {int(idx): pair[-1] for idx, pair in seg2label.items()}
        else:
            seg2label = None

        label2seg_refs = []
        for i in range(ref_length):
            if self.data_list[index]["index_refs"] is not None:
                index_ref = self.data_list[index]["index_refs"][i]
                seg2label_ref = load_json(index_ref)
                seg2label_ref = {int(idx): pair[-1] for idx, pair in seg2label_ref.items()}
                label2seg_ref = {label_index: seg_idx for seg_idx, label_index in seg2label_ref.items()}
            else:
                label2seg_ref = None
            label2seg_refs.append(label2seg_ref)

        # autmentation
        if self.augmentor is not None:
            line, seg, hint = self.augmentor(line, seg, hint)

            for i, (line_ref, seg_ref, gt_ref) in enumerate(zip(line_refs, seg_refs, gt_refs)):
                line_ref, seg_ref, gt_ref = self.augmentor(line_ref, seg_ref, gt_ref)
                line_refs[i] = line_ref
                seg_refs[i] = seg_ref
                gt_refs[i] = gt_ref

        mask_hair = (hint == 1)  # hair index
        mask_skin = (hint == 2)  # skin index
        mask_other = (hint == 3) # other index
        parse_mask = np.stack([mask_hair, mask_skin, mask_other], axis=0).astype(np.float32)

        # if self.random_color_hint:
        #     colors = generate_random_colors(3)
        #     hint[mask_hair] = colors[0]
        #     hint[mask_skin] = colors[1]
        #     hint[mask_other] = colors[2]

        seg_list = np.unique(seg[seg != 0])
        seg_list_refs = [np.unique(seg_ref[seg_ref != 0]) for seg_ref in seg_refs]

        if self.shuffle_label:
            np.random.shuffle(seg_list)
            for seg_list_ref in seg_list_refs:
                np.random.shuffle(seg_list_ref)

        seg_match_idxes = []
        for seg_list_ref in seg_list_refs:
            ref2pos = {ref_idx: i for i, ref_idx in enumerate(seg_list_ref)}
            if seg2label is not None:
                seg_match_idx = [ref2pos[seg2label[seg_idx]] if seg2label[seg_idx] in ref2pos else -1 for seg_idx in seg_list]
            else:
                seg_match_idx = [ref2pos[seg_idx] if seg_idx in ref2pos else -1 for seg_idx in seg_list]
            seg_match_idxes.append(seg_match_idx)
        seg_match_idxes = np.array(seg_match_idxes).astype(np.int64)

        h, w = seg.shape
        hh = np.arange(h)
        ww = np.arange(w)
        xx, yy = np.meshgrid(ww, hh)

        kpts = []
        # cpts = []
        numpts = []
        seg_tags = []
        # seg_tag_indices = []
        used_tags = []
        seg_relabeled = np.zeros_like(seg)
        # tag_mask = np.zeros_like(seg)

        for i, seg_idx in enumerate(seg_list):
            mask = seg == seg_idx

            xs = xx[mask]
            ys = yy[mask]
            xmin = xs.min()
            xmax = xs.max()
            ymin = ys.min()
            ymax = ys.max()
            # xmean = xs.mean()
            # ymean = ys.mean()
            kpts.append([xmin, xmax, ymin, ymax])
            # cpts.append([xmean, ymean])
            numpts.append(mask.sum())

            tag = colorbook.get_color_name(color_map[seg_idx])
            tag = tag.split()[-1]
            seg_tags.append(tag)
            if tag not in used_tags:
                used_tags.append(tag)

            seg_relabeled[mask] = i + 1  # 0 is for black line, start from 1
            # tag_idx = used_tags.index(tag)
            # tag_mask[mask] = tag_idx + 1
            # seg_tag_indices.append(tag_idx)

        kpts = np.stack(kpts)
        # cpts = np.stack(cpts)
        numpts = np.stack(numpts)

        kpts_refs = []
        # cpts_refs = []
        # numpts_refs = []
        seg_tags_refs = []
        # seg_tag_indices_refs = []
        seg_relabeled_refs = []
        tag_mask_refs = []
        # colorized_refs = []

        used_tags_ref = []
        tag_freq_ref = Counter()

        for seg_ref, seg_list_ref, color_map_ref, label2seg_ref in zip(seg_refs, seg_list_refs, color_map_refs, label2seg_refs):

            kpts_ref = []
            # cpts_ref = []
            numpts_ref = []
            seg_tags_ref = []
            # seg_tag_indices_ref = []
            seg_relabeled_ref = np.zeros_like(seg_ref)
            tag_mask_ref = np.zeros_like(seg_ref)
            # colorized_ref = np.zeros((*seg_ref.shape, 3))

            for i, ref_idx in enumerate(seg_list_ref):
                mask = seg_ref == ref_idx

                xs = xx[mask]
                ys = yy[mask]
                xmin = xs.min()
                xmax = xs.max()
                ymin = ys.min()
                ymax = ys.max()
                # xmean = xs.mean()
                # ymean = ys.mean()
                kpts_ref.append([xmin, xmax, ymin, ymax])
                # cpts_ref.append([xmean, ymean])
                numpts_ref.append(mask.sum())

                if label2seg_ref is not None:
                    seg_idx = label2seg_ref.get(ref_idx, 1)
                else:
                    seg_idx = ref_idx
                tag = colorbook.get_color_name(color_map_ref[seg_idx])
                tag = tag.split()[-1]
                seg_tags_ref.append(tag)
                if tag not in used_tags_ref:
                    used_tags_ref.append(tag)

                seg_relabeled_ref[mask] = i + 1  # 0 is for black line, start from 1
                tag_idx_ref = used_tags_ref.index(tag)
                tag_mask_ref[mask] = tag_idx_ref + 1
                # seg_tag_indices_ref.append(tag_idx_ref)
                # colorized_ref[mask] = default_colorbook[tag]

            kpts_refs.append(np.stack(kpts_ref))
            # cpts_refs.append(np.stack(cpts_ref))
            # numpts_refs.append(np.stack(numpts_ref))
            seg_tags_refs.append(seg_tags_ref)
            # seg_tag_indices_refs.append(seg_tag_indices_ref)
            seg_relabeled_refs.append(seg_relabeled_ref)
            tag_mask_refs.append(tag_mask_ref)
            # colorized_refs.append(colorized_ref)
            tag_freq_ref.update(Counter(np.unique(tag_mask_ref)))

        # seg_tag_indices = np.array(seg_tag_indices).astype(np.uint8)
        # seg_tag_indices_refs = np.array(seg_tag_indices_refs).astype(np.uint8)
        tag_freq_ref = [tag_freq_ref[i] for i in range(len(used_tags_ref))]
        tag_freq_ref = np.array(tag_freq_ref).astype(np.uint8)
        tag_match_idx = [used_tags_ref.index(seg_tag) if seg_tag in used_tags_ref else -1 for seg_tag in seg_tags]
        tag_match_idx = np.array(tag_match_idx).astype(np.int64)

        seg = seg_relabeled
        seg_refs = seg_relabeled_refs

        parse_mask_refs = []
        colored_tag_mask_refs = []
        colors = generate_random_colors(len(used_tags_ref))
        colors = np.concatenate([np.array([[0, 0, 0]]), colors], axis=0)
        tags = ["line"] + used_tags_ref
        for tag_mask_ref in tag_mask_refs:
            colored_tag_mask_ref = np.ones([*tag_mask_ref.shape[:2], 3]) * 255
            for i, tag in enumerate(tags):
                if tag == "background":
                    continue
                mask = tag_mask_ref == i
                colored_tag_mask_ref[mask] = colors[i]
            colored_tag_mask_refs.append(colored_tag_mask_ref)

            parse_mask_ref = np.zeros([3, *tag_mask_ref.shape[:2]]).astype(np.float32)
            for i, tag in enumerate(tags):
                if tag in ["background", "eye", "mouth"]:
                    continue
                mask = tag_mask_ref == i
                if tag == "hair":
                    parse_mask_ref[0][mask] = 1.0
                elif tag == "skin":
                    parse_mask_ref[1][mask] = 1.0
                else:  # other
                    parse_mask_ref[2][mask] = 1.0
            parse_mask_refs.append(parse_mask_ref)

        # np to tensor
        numpts = torch.from_numpy(numpts)[None]
        # numpts_refs = [torch.from_numpy(numpts_ref)[None] for numpts_ref in numpts_refs]
        line = torch.from_numpy(line).permute(2, 0, 1).float() / 255.0
        line_refs = [torch.from_numpy(line_ref).permute(2, 0, 1).float() / 255.0 for line_ref in line_refs]
        seg = torch.from_numpy(seg)[None]
        seg_refs = [torch.from_numpy(seg_ref)[None] for seg_ref in seg_refs]
        # tag_mask = torch.from_numpy(tag_mask)[None]
        tag_mask_refs = [torch.from_numpy(tag_mask_ref)[None] for tag_mask_ref in tag_mask_refs]
        colored_tag_mask_refs = [torch.from_numpy(colored_tag_mask_ref).permute(2, 0, 1).float() / 255.0 for colored_tag_mask_ref in colored_tag_mask_refs]
        # seg_tag_indices_refs = [torch.from_numpy(seg_tag_indices_ref) for seg_tag_indices_ref in seg_tag_indices_refs]
        # colorized_refs = [torch.from_numpy(colorized_ref).permute(2, 0, 1).float() / 255.0 for colorized_ref in colorized_refs]
        tag_freq_ref = torch.from_numpy(tag_freq_ref)
        tag_match_idx = torch.from_numpy(tag_match_idx).float()
        seg_match_idxes = [torch.from_numpy(seg_match_idx).float() for seg_match_idx in seg_match_idxes]
        #hint = torch.from_numpy(hint).permute(2, 0, 1).float() / 255.0
        mask_hair = torch.from_numpy(mask_hair).int()[None]
        mask_skin = torch.from_numpy(mask_skin).int()[None]
        mask_other = torch.from_numpy(mask_other).int()[None]

        # if self.split in ["train", "val"]:
        #     colored_img_refs = [recolorize_seg(seg_ref) if self.random_color_img else torch.tensor([0]) for seg_ref in seg_refs]
        # elif self.split == "test":
        #     gt_refs = [read_img_2_np(gt_ref) for gt_ref in self.data_list[index]["gt_refs"]]
        #     colored_img_refs = [recolorize_gt(gt_ref) if self.random_color_img else torch.tensor([0]) for gt_ref in gt_refs]

        colored_gt_refs = [recolorize_gt(gt_ref) for gt_ref in gt_refs]
        # colored_seg_refs = [recolorize_seg(seg_ref) if self.random_color_img else torch.tensor([0]) for seg_ref in seg_refs]

        return {
            "ref_length": ref_length,
            "file_name": file_name,
            # "file_name_refs": file_name_refs[0] if len(file_name_refs) == 1 else file_name_refs,
            "keypoints": kpts,
            "keypoints_refs": kpts_refs[0] if len(kpts_refs) == 1 else kpts_refs,
            # "centerpoints": cpts,
            # "centerpoints_refs": cpts_refs[0] if len(cpts_refs) == 1 else cpts_refs,
            "numpts": numpts,
            # "numpts_refs": numpts_refs[0] if len(numpts_refs) == 1 else numpts_refs,
            "seg_tags": seg_tags,
            "seg_tags_refs": seg_tags_refs[0] if len(seg_tags_refs) == 1 else seg_tags_refs,
            "line": line,
            "line_refs": line_refs[0] if len(line_refs) == 1 else line_refs,
            "segment": seg,
            "segment_refs": seg_refs[0] if len(seg_refs) == 1 else seg_refs,
            # "colored_img_refs": colored_img_refs[0] if len(colored_img_refs) == 1 else colored_img_refs,
            "colored_gt_refs": colored_gt_refs[0] if len(colored_gt_refs) == 1 else colored_gt_refs,
            # "tag_mask": tag_mask,
            "tag_mask_refs": tag_mask_refs[0] if len(tag_mask_refs) == 1 else tag_mask_refs,
            "colored_tag_mask_refs": colored_tag_mask_refs[0] if len(colored_tag_mask_refs) == 1 else colored_tag_mask_refs,
            # "colorized_refs": colorized_refs[0] if len(colorized_refs) == 1 else colorized_refs,
            # "used_tags": used_tags,
            "used_tags_ref": used_tags_ref,
            # "seg_tag_indices": seg_tag_indices,  # TODO
            # "seg_tag_indices_refs": seg_tag_indices_refs[0] if len(seg_tag_indices_refs) == 1 else seg_tag_indices_refs,  # TODO
            "tag_freq_ref": tag_freq_ref,  # occur times among all refs
            "seg_list_refs": seg_list_refs[0] if len(seg_list_refs) == 1 else seg_list_refs,
            "tag_all_matches": tag_match_idx,
            "seg_all_matches": seg_match_idxes[0] if len(seg_match_idxes) == 1 else seg_match_idxes,
            # "hint": hint,
            "mask_hair": mask_hair,
            "mask_skin": mask_skin,
            "mask_other": mask_other,
            "parse_mask": parse_mask,
            "parse_mask_refs": parse_mask_refs[0] if len(parse_mask_refs) == 1 else parse_mask_refs,
        }

    def __rmul__(self, v):
        self.data_list = v * self.data_list
        return self

    def __len__(self):
        return len(self.data_list)


@DATASET_REGISTRY.register()
class PaintBucketRefTagSegDataset(AnimeTagSegDataset):
    def __init__(self, opt):
        aug_params = opt["aug_params"] if "aug_params" in opt else None
        super(PaintBucketRefTagSegDataset, self).__init__(aug_params)

        self.opt = opt
        self.root = opt["root"]
        self.shuffle_label = opt["shuffle_label"] if "shuffle_label" in opt else False
        self.use_ref_for_train = opt["use_ref_for_train"] if "use_ref_for_train" in opt else False
        self.num_per_character = opt["num_per_character"] if "num_per_character" in opt else None

        self.line_name = opt["line_name"] if "line_name" in opt else "line"
        self.hint_name = opt["hint_name"] if "hint_name" in opt else None

        self.skip_character = opt["skip_character"] if "skip_character" in opt else None
        if not isinstance(self.skip_character, list):
            self.skip_character = [self.skip_character]

        self.split = opt["split"] if "split" in opt else None
        assert self.split in ["train", "val", "test"]
        if self.split != "train":
            self.augmentor = None
            self.shuffle_label = False

        if self.split == "val":
            self.val_length = opt["val_length"] if "val_length" in opt else 50

        for character in os.listdir(self.root):
            if character in self.skip_character:
                continue
            cnt = 0

            colorbook_root = osp.join(self.root, character, "colorbook.yml")
            gt_root = osp.join(self.root, character, "gt")
            gt_ref_root = osp.join(self.root, character, "ref", "gt")
            line_root = osp.join(self.root, character, self.line_name)
            line_ref_root = osp.join(self.root, character, "ref", "line")
            seg_root = osp.join(self.root, character, "seg")
            seg_ref_root = osp.join(self.root, character, "ref", "seg")
            if self.split in ["train", "val"]:
                seg_ref_root = seg_ref_root.replace("seg", "label")
                color_root = osp.join(self.root, character, "json_color")
                color_ref_root = osp.join(self.root, character, "ref", "json_color")
                index_root = osp.join(self.root, character, "json_index")
                index_ref_root = osp.join(self.root, character, "ref", "json_index")
            elif self.split == "test":
                color_root = seg_root
                color_ref_root = seg_ref_root
                index_root = None
                index_ref_root = None

            gt_list = sorted(glob(osp.join(gt_root, "*.png")))
            line_list = sorted(glob(osp.join(line_root, "*.png")))
            seg_list = sorted(glob(osp.join(seg_root, "*.png")))
            color_list = sorted(glob(osp.join(color_root, "*.json")))
            if index_root:
                index_list = sorted(glob(osp.join(index_root, "*.json")))

            gt_ref_list = sorted(glob(osp.join(gt_ref_root, "*.png")))
            line_ref_list = sorted(glob(osp.join(line_ref_root, "*.png")))
            seg_ref_list = sorted(glob(osp.join(seg_ref_root, "*.png")))
            color_ref_list = sorted(glob(osp.join(color_ref_root, "*.json")))
            if index_ref_root:
                index_ref_list = sorted(glob(osp.join(index_ref_root, "*.json")))

            if self.hint_name is not None:
                hint_root = osp.join(self.root, character, self.hint_name)
                hint_list = sorted(glob(osp.join(hint_root, "*.png")))

            L = len(line_list)

            if self.split == "train" and not self.use_ref_for_train:
                for i in range(L):
                    ref_idx = np.random.randint(L)
                    data_sample = {
                        "ref_length": 1,
                        "colorbook": colorbook_root,
                        "file_name": line_list[i][:-4],
                        # "gt": gt_list[i],
                        "line": line_list[i],
                        "hint": hint_list[i] if self.hint_name is not None else None,
                        "seg": seg_list[i],
                        "color": color_list[i],
                        "index": index_list[i],
                        "file_name_refs": [line_list[ref_idx][:-4]],
                        "gt_refs": [gt_list[ref_idx]],
                        "line_refs": [line_list[ref_idx]],
                        "seg_refs": [seg_list[ref_idx].replace("seg", "label")],
                        "color_refs": [color_list[ref_idx]],
                        "index_refs": [index_list[ref_idx]],
                    }
                    self.data_list += [data_sample]
                    cnt += 1
                    if self.num_per_character and cnt == self.num_per_character:
                        break
            elif self.split == "val":
                for i in random.sample(range(L), self.val_length):
                    data_sample = {
                        "ref_length": 1,
                        "colorbook": colorbook_root,
                        "file_name": line_list[i][:-4],
                        "line": line_list[i],
                        "hint": hint_list[i] if self.hint_name is not None else None,
                        "seg": seg_list[i],
                        "color": color_list[i],
                        "index": index_list[i],
                        "file_name_refs": [line_ref_list[0][:-4]],
                        "gt_refs": [gt_ref_list[0]],
                        "line_refs": [line_ref_list[0]],
                        "seg_refs": [seg_ref_list[0]],
                        "color_refs": [color_ref_list[0]],
                        "index_refs": [index_ref_list[0]],
                    }
                    self.data_list += [data_sample]
                    cnt += 1
                    if self.num_per_character and cnt == self.num_per_character:
                        break
            else:  # 1. split: test  2. split=train and use_ref_for_train=true
                for i in range(L):
                    ref_idx = np.random.randint(4) if self.split == "train" else 0
                    data_sample = {
                        "ref_length": 1,
                        "colorbook": colorbook_root,
                        "file_name": line_list[i][:-4],
                        "line": line_list[i],
                        "hint": hint_list[i] if self.hint_name is not None else None,
                        "seg": seg_list[i],
                        "color": color_list[i],
                        "index": index_list[i] if index_root else None,
                        "file_name_refs": [line_ref_list[ref_idx][:-4]],
                        "gt_refs": [gt_ref_list[ref_idx]],
                        "line_refs": [line_ref_list[ref_idx]],
                        "seg_refs": [seg_ref_list[ref_idx]],
                        "color_refs": [color_ref_list[ref_idx]],
                        "index_refs": [index_ref_list[ref_idx]] if index_ref_root else None,
                    }
                    self.data_list += [data_sample]
                    cnt += 1
                    if self.num_per_character and cnt == self.num_per_character:
                        break

        print("Length of data sample list is", len(self.data_list))


@DATASET_REGISTRY.register()
class PaintBucketMultiRefTagSegDataset(AnimeTagSegDataset):
    def __init__(self, opt):
        aug_params = opt["aug_params"] if "aug_params" in opt else None
        super(PaintBucketMultiRefTagSegDataset, self).__init__(aug_params)

        self.opt = opt
        self.root = opt["root"]
        self.shuffle_label = opt["shuffle_label"] if "shuffle_label" in opt else False
        self.use_ref_for_train = opt["use_ref_for_train"] if "use_ref_for_train" in opt else False
        self.num_of_reference = opt["num_of_reference"] if "num_of_reference" in opt else None
        self.num_per_character = opt["num_per_character"] if "num_per_character" in opt else None

        self.line_name = opt["line_name"] if "line_name" in opt else "line"
        self.hint_name = opt["hint_name"] if "hint_name" in opt else None

        self.skip_character = opt["skip_character"] if "skip_character" in opt else None
        if not isinstance(self.skip_character, list):
            self.skip_character = [self.skip_character]

        self.split = opt["split"] if "split" in opt else None
        assert self.split in ["train", "val", "test"]
        if self.split != "train":
            self.augmentor = None
            self.shuffle_label = False

        if self.split == "val":
            self.val_length = opt["val_length"] if "val_length" in opt else 50

        N = self.num_of_reference or 4

        for character in os.listdir(self.root):
            if character in self.skip_character:
                continue
            cnt = 0

            colorbook_root = osp.join(self.root, character, "colorbook.yml")
            gt_ref_root = osp.join(self.root, character, "ref", "gt")
            line_root = osp.join(self.root, character, "line")
            line_ref_root = osp.join(self.root, character, "ref", "line")
            seg_root = osp.join(self.root, character, "seg")
            seg_ref_root = osp.join(self.root, character, "ref", "seg")
            if self.split in ["train", "val"]:
                seg_ref_root = seg_ref_root.replace("seg", "label")
                color_root = osp.join(self.root, character, "json_color")
                color_ref_root = osp.join(self.root, character, "ref", "json_color")
                index_root = osp.join(self.root, character, "json_index")
                index_ref_root = osp.join(self.root, character, "ref", "json_index")
            elif self.split == "test":
                color_root = seg_root
                color_ref_root = seg_ref_root
                index_root = None
                index_ref_root = None

            line_list = sorted(glob(osp.join(line_root, "*.png")))
            seg_list = sorted(glob(osp.join(seg_root, "*.png")))
            color_list = sorted(glob(osp.join(color_root, "*.json")))
            index_list = sorted(glob(osp.join(index_root, "*.json"))) if index_root else None

            gt_ref_list = sorted(glob(osp.join(gt_ref_root, "*.png")))
            line_ref_list = sorted(glob(osp.join(line_ref_root, "*.png")))
            seg_ref_list = sorted(glob(osp.join(seg_ref_root, "*.png")))
            color_ref_list = sorted(glob(osp.join(color_ref_root, "*.json")))
            index_ref_list = sorted(glob(osp.join(index_ref_root, "*.json"))) if index_ref_root else None

            if self.hint_name is not None:
                hint_root = osp.join(self.root, character, self.hint_name)
                hint_list = sorted(glob(osp.join(hint_root, "*.png")))

            L = len(line_list)

            if self.split == "train" and not self.use_ref_for_train:
                for i in range(L):
                    ref_idxes = random.sample(range(L), N)
                    data_sample = {
                        "ref_length": N,
                        "colorbook": colorbook_root,
                        "file_name": line_list[i][:-4],
                        "line": line_list[i],
                        "hint": hint_list[i] if self.hint_name is not None else None,
                        "seg": seg_list[i],
                        "color": color_list[i],
                        "index": index_list[i],
                        "file_name_refs": [line_list[idx][:-4] for idx in ref_idxes],
                        "line_refs": [line_list[idx] for idx in ref_idxes],
                        "seg_refs": [seg_list[idx].replace("seg", "label") for idx in ref_idxes],
                        "color_refs": [color_list[idx] for idx in ref_idxes],
                        "index_refs": [index_list[idx] for idx in ref_idxes],
                    }
                    self.data_list += [data_sample]
                    cnt += 1
                    if self.num_per_character and cnt == self.num_per_character:
                        break
            elif self.split == "val":
                for i in random.sample(range(L), self.val_length):
                    data_sample = {
                        "ref_length": N,
                        "colorbook": colorbook_root,
                        "file_name": line_list[i][:-4],
                        "line": line_list[i],
                        "hint": hint_list[i] if self.hint_name is not None else None,
                        "seg": seg_list[i],
                        "color": color_list[i],
                        "index": index_list[i],
                        "file_name_refs": [line_ref[:-4] for line_ref in line_ref_list[:N]],
                        "gt_refs": gt_ref_list[:N],
                        "line_refs": line_ref_list[:N],
                        "seg_refs": seg_ref_list[:N],
                        "color_refs": color_ref_list[:N],
                        "index_refs": index_ref_list[:N],
                    }
                    self.data_list += [data_sample]
                    cnt += 1
                    if self.num_per_character and cnt == self.num_per_character:
                        break
            else:  # 1. split=test  2. split=train and use_ref_for_train=true
                for i in range(L):
                    data_sample = {
                        "ref_length": N,
                        "colorbook": colorbook_root,
                        "file_name": line_list[i][:-4],
                        "line": line_list[i],
                        "hint": hint_list[i] if self.hint_name is not None else None,
                        "seg": seg_list[i],
                        "color": color_list[i],
                        "index": index_list[i] if index_list else None,
                        "file_name_refs": [line_ref[:-4] for line_ref in line_ref_list[:N]],
                        "gt_refs": gt_ref_list[:N],
                        "line_refs": line_ref_list[:N],
                        "seg_refs": seg_ref_list[:N],
                        "color_refs": color_ref_list[:N],
                        "index_refs": index_ref_list[:N] if index_ref_list else None,
                    }
                    self.data_list += [data_sample]
                    cnt += 1
                    if self.num_per_character and cnt == self.num_per_character:
                        break

        print("Length of data sample list is", len(self.data_list))
