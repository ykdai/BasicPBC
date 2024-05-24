import cv2
import numpy as np
import os
from skimage import measure, morphology

from linefiller.linefiller.thinning import thinning
from linefiller.linefiller.trappedball_fill import build_fill_map, flood_fill_multi, mark_fill, merge_fill, show_fill_map, trapped_ball_fill_multi
from paint.utils import generate_random_colors, np_2_labelpng, read_line_2_np


class LineArt:
    def __init__(self, lineart_img, colorbook=None, new_colorbook=None):
        self.colorbook = colorbook
        if new_colorbook is not None:
            self.new_colorbook = new_colorbook
        else:
            self.new_colorbook = colorbook
        self.lineart = lineart_img
        self.alpha = lineart_img[:, :, 3]
        # binarize alpha channel
        self.line_all = self.alpha < 127
        self.line_black = np.all(self.lineart == [0, 0, 0, 255], axis=-1)
        # Converte binarized alpha channel to label image
        self.label_img = measure.label(self.line_all, connectivity=1)
        self.erase_single_pixels(3)
        self.label_hightlight_shadow = [1] * (self.label_img.max() + 1)  # 0 means highlight, 1 means normal and 2/3 means shadow1/2

    def relabel(self):
        num_labels = self.label_img.max() + 1
        index = 1
        for i in range(1, num_labels):
            if i in self.label_img:
                self.label_img[self.label_img == i] = index
                index += 1

    def erase_single_pixels(self, threshold=2):
        # Remove all regions with pixels less than threshold
        props = measure.regionprops(self.label_img)
        selem = morphology.square(3)
        # print(self.label_img.max()+1)
        for i in range(1, self.label_img.max() + 1):
            if props[i - 1].area <= threshold:
                mask = self.label_img == i
                dilated_mask = morphology.binary_dilation(mask, selem) & ~mask
                common_labels = np.unique(self.label_img[dilated_mask])
                common_labels = [x for x in common_labels if x != 0]

                region_props = [props[label - 1] for label in common_labels]
                if len(region_props) == 0:
                    min_area_label = 0
                else:
                    min_area_label = min(region_props, key=lambda x: x.area).label
                self.label_img[mask] = min_area_label
        self.relabel()

    def label_color_line(self):
        # Colorize red/blue/green lines, find the nearest pixels around the color line, then merge the label
        # In real animation production, animators will colorize the highlight and shadow first.
        # We follow this pipeline to process color line first.

        # line_mask=1-self.line_all
        # line_color=line_mask>self.line_black
        line_colortype_list = [[255, 0, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255]]
        props = measure.regionprops(self.label_img)
        props_max = len(props)
        # print(len(props))
        for colortype in line_colortype_list:
            line_color = np.all(self.lineart == colortype, axis=-1)
            label_color_line = measure.label(line_color, connectivity=2)
            for i in range(1, label_color_line.max() + 1):
                mask = label_color_line == i
                dilated_lines = morphology.binary_dilation(mask) & ~mask
                # Find the labels that appear in both the dilated lines and the label_img
                common_labels = np.unique(self.label_img[dilated_lines])
                common_labels = [x for x in common_labels if x != 0]
                # Get the properties of each white region that appears in the dilated lines and the label_img
                if len(common_labels) != 0:
                    region_props = [props[label - 1] for label in common_labels if label - 1 < props_max]
                else:
                    region_props = [props[0]]
                # Find the label of the white region with the largest area
                if len(region_props) == 0:
                    min_area_label = 0
                    self.label_img[mask] = min_area_label
                else:
                    min_area_label = min(region_props, key=lambda x: x.area).label
                    if len(region_props) > 1:
                        if colortype[0] == 255:  # R color represents highlight
                            self.label_hightlight_shadow[min_area_label] = 0
                        elif colortype[2] == 255:  # B color represents shadow
                            self.label_hightlight_shadow[min_area_label] = 2
                        self.label_img[mask] = min_area_label
                        # print("label color line!")
                    elif len(region_props) == 1:
                        # Line or dot shaped highlight or shadow, add new label
                        self.label_img[mask] = self.label_img.max() + 1
                        self.label_hightlight_shadow.append(0)

    def colorize_based_ref(self, ref_img, color_type="all"):
        # Output a colorized label image based on a reference image ref_img
        ref_img = ref_img[:, :, :3]
        colorized_img = np.zeros_like(self.lineart[:, :, :3])
        for i in range(1, self.label_img.max() + 1):
            mask = self.label_img == i
            mean_rgb = np.sum(ref_img[mask], axis=0) / np.sum(np.array(mask))
            nearest_color = self.colorbook.find_nearest_color(mean_rgb, color_type)
            colorized_img[mask] = nearest_color
        return colorized_img

    def colorize_random(self):
        # Output a colorized label image
        colorized_img = np.zeros_like(self.lineart[:, :, :3])
        color_list = generate_random_colors(self.label_img.max() + 1)
        for i in range(1, self.label_img.max() + 1):
            colorized_img[self.label_img == i] = color_list[i]
        return colorized_img

    def save_label_image(self, save_path, format="png"):
        # Output the label image at the save_path
        # Black line is marked as '0' and other regions start from '1'
        try:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # Save the label_image to .npy file
            if format == "npy":
                np.save(save_path, self.label_img)
            elif format == "png":
                np_2_labelpng(self.label_img, save_path)
            else:
                assert False, "This format is not supported."
        except Exception as e:
            print(f"Error while saving the label image: {e}")


def trappedball_fill(img_path, save_path, radius=4, contour=False):

    im = read_line_2_np(img_path, channel=3)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY)

    fills = []
    result = binary
    radius = max(4, radius)

    fill = trapped_ball_fill_multi(result, radius, method="max")
    fills += fill
    result = mark_fill(result, fill)

    fill = trapped_ball_fill_multi(result, radius // 2, method=None)
    fills += fill
    result = mark_fill(result, fill)

    fill = trapped_ball_fill_multi(result, 1, method=None)
    fills += fill
    result = mark_fill(result, fill)

    fill = flood_fill_multi(result)
    fills += fill

    fillmap = build_fill_map(result, fills)
    # cv2.imwrite("tmp/fills.png", show_fill_map(fillmap))

    fillmap = merge_fill(fillmap)

    if contour:
        cv2.imwrite(save_path, show_fill_map(fillmap))
    else:
        cv2.imwrite(save_path, show_fill_map(thinning(fillmap)))
