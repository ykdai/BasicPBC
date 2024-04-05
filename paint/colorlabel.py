import copy
import numpy as np
import os
from glob import glob
from skimage import color, io, measure, morphology
from skimage.measure import label
from skimage.morphology import binary_dilation, square
from tqdm import tqdm

from paint.lineart import LineArt
from paint.utils import dump_json, labelpng_2_np, np_2_labelpng


class ColorLabel:
    # This Module is mainly for label image
    def __init__(self):
        self.colorbook = None

    def load_colorbook(self, colorbook):
        # Load colorbook
        self.colorbook = colorbook

    def extract_black_line(self, label_img, save_path=None, erosion_flag=True):
        # label img should be loaded by using skimage
        label_np = labelpng_2_np(label_img)
        line = np.ones_like(label_img) * [255, 255, 255, 0]
        line_black = label_np == 0  # black line in (H,W)
        unique_labels = np.unique(label_np)
        for label in unique_labels:
            if label != 0:
                mask = label_np == label
                if not erosion_flag:
                    added_lines = morphology.binary_dilation(mask) & ~mask
                else:
                    added_lines = mask & ~morphology.binary_erosion(mask | line_black)
                line_black += added_lines
        line_mask = line_black >= 1
        line[line_mask] = [0, 0, 0, 255]  # Change the background of 0 alpha region to white. It is designed for some softwares such as PaintMan.
        if save_path is not None:
            io.imsave(save_path, line.astype(np.uint8))
        return line.astype(np.uint8)

    def generate_paired_data(self, color_img_path, label_img_path):
        label_img = io.imread(label_img_path)  # H x W
        label_np = labelpng_2_np(label_img)
        color_img = io.imread(color_img_path)

        color_dict = {}  # {'1': [255,233,255,0]}
        index_dict = {}  # {'1':[area,label]}

        line_black = self.extract_black_line(label_img)  # H x W x 4
        lineart = LineArt(line_black)
        index_img = lineart.label_img
        seg = np_2_labelpng(index_img)

        gt = copy.deepcopy(color_img)
        gt[np.all(line_black == [0, 0, 0, 255], axis=-1)] = [0, 0, 0, 255]

        props = measure.regionprops(index_img)
        for i in range(1, index_img.max() + 1):

            pos = props[i - 1].coords[0]

            index_color = color_img[pos[0], pos[1], :]
            index_label = label_np[pos[0], pos[1]]

            index_dict[str(i)] = [int(props[i - 1].area), int(index_label)]
            color_dict[str(i)] = index_color.tolist()

        return {"gt": gt, "line_black": line_black, "seg": seg, "json_color": color_dict, "json_index": index_dict}

    def process_folder(self, load_folder, save_folder):
        # Generate line for training data.
        # A folder 'gt' and 'label' should be under the path of load_folder
        os.makedirs(os.path.join(save_folder, "gt"), exist_ok=True)  # Use default setting to avoid deleting previous files
        os.makedirs(os.path.join(save_folder, "line_black"), exist_ok=True)
        os.makedirs(os.path.join(save_folder, "seg"), exist_ok=True)
        os.makedirs(os.path.join(save_folder, "json_color"), exist_ok=True)
        os.makedirs(os.path.join(save_folder, "json_index"), exist_ok=True)
        color_img_list = sorted(glob(os.path.join(load_folder, "gt_color", "*.png")))
        if len(color_img_list) > 0:
            for i, color_img_path in tqdm(enumerate(color_img_list)):
                label_img_path = color_img_path.replace("gt_color", "label")
                paired_data = self.generate_paired_data(color_img_path, label_img_path)
                io.imsave(os.path.join(save_folder, "gt/", str(i).zfill(4) + ".png"), paired_data["gt"], check_contrast=False)
                io.imsave(os.path.join(save_folder, "line_black/", str(i).zfill(4) + ".png"), paired_data["line_black"], check_contrast=False)
                io.imsave(os.path.join(save_folder, "seg/", str(i).zfill(4) + ".png"), paired_data["seg"], check_contrast=False)
                dump_json(paired_data["json_color"], os.path.join(save_folder, "json_color/", str(i).zfill(4) + ".json"))
                dump_json(paired_data["json_index"], os.path.join(save_folder, "json_index/", str(i).zfill(4) + ".json"))
        else:
            assert False, "Error! No images are loaded."

    def relabel_image(self, label_img, color_dict):
        # relabel the image based on the left-up pixel
        unique_labels = np.unique(label_img)
        relabeled_img = np.zeros_like(label_img)
        recolored_dict = {}

        label_mapping = {}
        new_label = 1  # 0 means the line art, we ignore the zero
        label_leftup_pix_dict = {}
        for label in unique_labels:
            if label != 0:
                mask = label_img == label
                coords = np.argwhere(mask)
                top_left_corner = coords.min(axis=0)  # 获取掩码最左上角的坐标
                label_leftup_pix_dict[label] = top_left_corner
        sorted_labels = sorted(label_leftup_pix_dict, key=lambda k: label_leftup_pix_dict[k][0])

        for label in sorted_labels:
            if label != 0:
                relabeled_img[label_img == label] = new_label
                label_mapping[label] = new_label
                recolored_dict[str(new_label)] = color_dict[str(label)]
                new_label += 1
        return relabeled_img, recolored_dict

    def extract_label_map(self, color_img_path, img_save_path=None, line_img_path=None, extract_seg=False):
        # This part is mainly for the test data.
        # This function will extract the label map for each colorized img.
        # Every connected segment will be viewed as one label.
        # If the extract_seg is true, this module will extract the segment rather than label
        img = io.imread(color_img_path)
        img = np.array(img)
        if line_img_path is not None:
            line = np.array(io.imread(line_img_path))
        labeled_img = np.zeros(img.shape[:2], dtype=np.int32)
        color_dict = {}
        neighborhood = square(3)
        index = 0  # index 0 means the black line.
        if self.colorbook is not None:
            color_list = self.colorbook.all_color_list
        else:
            img_data = img[:, :, :3]
            colors = img_data.reshape(-1, img_data.shape[-1])
            color_list = list(np.unique(colors, axis=0))
            # Background is the most bright
        for i, color in enumerate(color_list):
            mask = np.all(img[:, :, :3] == color, axis=-1)
            if np.max(mask) != 0 and np.max(color) != 0:
                if not extract_seg:
                    expanded_mask = binary_dilation(mask, footprint=neighborhood)
                else:
                    expanded_mask = mask
                labeled_color_regions, num_labels = label(expanded_mask, connectivity=2, return_num=True)
                labeled_color_regions = labeled_color_regions * mask
                for region_label in range(1, num_labels + 1):
                    region_mask = labeled_color_regions == region_label
                    if np.sum(region_mask) > 1:
                        # We do not calculate accuracy for single pixel region.
                        index += 1
                        labeled_img[region_mask] = index

                        if np.min(color) == 255:
                            color_new = np.append(color, 0)
                        else:
                            color_new = np.append(color, 255)
                        color_dict[str(index)] = color_new.tolist()
                    else:
                        # fill the line art and gt's single pixel
                        if line_img_path is not None:
                            line[region_mask] = [0, 0, 0, 255]
                        img[region_mask] = [0, 0, 0, 255]
        # relabel all the given labels to avoid data leakage
        labeled_img, color_dict = self.relabel_image(labeled_img, color_dict)

        if img_save_path is not None:
            np_2_labelpng(labeled_img, img_save_path)
            # dump_json(color_dict, img_save_path.replace(".png", ".json"))
        if line_img_path is not None:
            io.imsave(line_img_path, line, check_contrast=False)
            io.imsave(color_img_path, img, check_contrast=False)
        return labeled_img

    def extract_label_folder(self, load_folder, save_folder, color_folder_name="gt", extract_seg=False):
        # If fill single pixel is true, it will
        os.makedirs(os.path.join(save_folder, "seg"), exist_ok=True)  # Use default setting to avoid deleting previous files
        color_img_list = sorted(glob(os.path.join(load_folder, color_folder_name, "*.png")))
        line_img_list = sorted(glob(os.path.join(load_folder, "line", "*.png")))
        if len(color_img_list) > 0:
            for i, color_img_path in tqdm(enumerate(color_img_list)):
                label_img_path = color_img_path.replace(color_folder_name, "seg")
                line_img_path = line_img_list[i]

                self.extract_label_map(color_img_path, label_img_path, line_img_path, extract_seg)
        else:
            assert False, "Error! No images are loaded."
