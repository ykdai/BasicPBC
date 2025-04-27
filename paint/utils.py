import csv
import json
import numpy as np
import os
import torch
from glob import glob
from PIL import Image
from skimage import io
from skimage.measure import label
from skimage.morphology import binary_dilation, square
from tqdm import tqdm
from skimage.restoration import inpaint

from paint.colorbook import ColorBook

default_colorbook = {
    "background": [255, 255, 255],
    "bag": [0, 0, 255],
    "belt": [0, 165, 0],
    "glasses": [255, 0, 0],
    "hair": [60, 60, 60],
    "socks": [0, 128, 128],
    "hat": [255, 222, 68],
    "mouth": [220, 160, 150],
    "clothes": [90, 80, 160],
    "eye": [230, 230, 230],
    "shoes": [134, 76, 57],
    "skin": [250, 210, 180],
}


def load_json(file_path):
    with open(file_path, "r") as f:
        file_dict = json.load(f)
    return file_dict


def dump_json(save_dict, save_path):
    file_json = json.dumps(save_dict)
    with open(save_path, "w") as f:
        f.write(file_json)


def np_2_labelpng(input_np, save_path=None):
    # Transfer a 1-ch label image (numpy format) into the 3-ch label png, it can cover the label with index less than 256^3
    # Input is a numpy and output is the image. R*256^2+G^256+B is the label for each pixel.
    if len(input_np.shape) != 2:
        raise ValueError("Input numpy array must be 2D label image.")

    # Convert the 1-ch label image (numpy format) into a 3-ch label png
    h, w = input_np.shape
    if np.min(input_np) == -1:
        output_img = np.zeros((h, w, 4), dtype=np.uint8)
        input_np_uint = input_np.astype(np.int32)
        output_img[:, :, 0] = (input_np_uint >> 16) & 255
        output_img[:, :, 1] = (input_np_uint >> 8) & 255
        output_img[:, :, 2] = input_np_uint & 255
        output_img[:, :, 3] = 255
        output_img[input_np == -1] = [255, 255, 255, 0]
    else:
        output_img = np.zeros((h, w, 3), dtype=np.uint8)
        input_np = input_np.astype(np.int32)
        output_img[:, :, 0] = (input_np >> 16) & 255
        output_img[:, :, 1] = (input_np >> 8) & 255
        output_img[:, :, 2] = input_np & 255

    # If save_path is provided, save the image to that path
    if save_path is not None:
        directory = os.path.dirname(save_path)
        if len(directory) != 0 and not os.path.exists(directory):
            os.makedirs(directory)
        io.imsave(save_path, output_img, check_contrast=False)

    return output_img


def labelpng_2_np(input_img):
    # Transfer a 3-ch or 4-ch label image into the 1-ch label image (numpy format).
    # Input is a image (skimage) and output is the np. R*256^2+G^256+B is the label for each pixel.
    if len(input_img.shape) == 3 and input_img.shape[2] == 3:
        # Convert the 3-ch label image into a 1-ch label image (numpy format)
        output_np = (input_img[:, :, 0].astype(np.int32) << 16) + (input_img[:, :, 1].astype(np.int32) << 8) + input_img[:, :, 2].astype(np.int32)
    elif len(input_img.shape) == 3 and input_img.shape[2] == 4:
        # For 4-ch label image transparent region is -1
        output_np = (input_img[:, :, 0].astype(np.int32) << 16) + (input_img[:, :, 1].astype(np.int32) << 8) + input_img[:, :, 2].astype(np.int32)
        output_np[input_img[:, :, 3] == 0] = -1
    else:
        raise ValueError("Input image must be a 3-ch or 4-ch image.")
    return output_np


def read_seg_2_np(seg_path, type=np.int64):
    seg = io.imread(seg_path)
    seg_np = labelpng_2_np(seg).astype(type)
    return seg_np


def read_img_2_np(img_path, channel=3, type=np.uint8):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img).astype(type)
    if len(img_np.shape) == 2:
        img_np = np.tile(img_np[..., None], (1, 1, channel))
    else:
        img_np = img_np[..., :channel]
    return img_np

def process_line_anno(line_np,seg_np,use_color=False):
    # extract the semantic information from the annotated line image
    hair_color=[[185,137,253],[239,238,52]]
    skin_color=[[255,227,185],[255,179,196]]
    other_color=[[188,201,251],[255,245,139]]

    hair_output_color = [250,0,0]#[252, 243, 253]
    skin_output_color = [0,250,0]#[253, 242, 229]
    other_output_color = [0,0,250]#[244, 252, 254]

    hair_mask = np.zeros_like(line_np[:,:,0])
    
    for color in hair_color:
        mask = np.all(line_np == color, axis=-1)
        hair_mask = hair_mask | mask

    skin_mask = np.zeros_like(line_np[:,:,0])
    for color in skin_color:
        mask = np.all(line_np == color, axis=-1)
        skin_mask = skin_mask | mask

    other_mask = np.zeros_like(line_np[:,:,0])
    for color in other_color:
        mask = np.all(line_np == color, axis=-1)
        other_mask = other_mask | mask
    
    if use_color:
        # return the colorized line image
        for i in range(1,seg_np.max()):
            mask = seg_np == i
            if np.sum(mask) == 0:
                continue
            if np.sum(mask & hair_mask) > 0:
                line_np[mask] = hair_output_color
            elif np.sum(mask & skin_mask) > 0:
                line_np[mask] = skin_output_color
            elif np.sum(mask & other_mask) > 0:
                line_np[mask] = other_output_color
        return line_np
    else:
        # return the line mask
        line_mask = np.zeros((line_np.shape[0],line_np.shape[1])) #It should be in [H,W]
        for i in range(1,seg_np.max()):
            mask = seg_np == i
            if np.sum(mask) == 0:
                continue
            if np.sum(mask & hair_mask) > 0:
                line_mask[mask] = 1
            elif np.sum(mask & skin_mask) > 0:
                line_mask[mask] = 2
            elif np.sum(mask & other_mask) > 0:
                line_mask[mask] = 3
        return line_mask

def generate_random_colors(num_colors, shuffle=True):
    if num_colors == 0:
        return np.array([]).astype(np.uint8)
    num_cubes = int(np.ceil(num_colors ** (1 / 3)))
    step = int(np.ceil(256 / num_cubes))
    num_colors_actual = num_cubes**3
    while num_colors_actual < num_colors:
        step += 1
        num_cubes = int(np.ceil(256 / step))
        num_colors_actual = num_cubes**3
    r, g, b = np.meshgrid(range(num_cubes), range(num_cubes), range(num_cubes))
    colors = np.stack([r.flatten(), g.flatten(), b.flatten()], axis=-1) * step + int(step / 2)
    if shuffle:
        np.random.shuffle(colors)
    return colors[:num_colors].astype(np.uint8)


def recolorize_seg(label_tensor, colorize_line=False):
    """
    Label tensor is in [H,W], ranges from [0,N]. 0 means the black line.
    Output tensor is also in [3,H,W], ranges from [0,1].
    """
    device = label_tensor.device

    recolorized = 255 * torch.ones((3, *label_tensor.size()[-2:])).int()
    labels = label_tensor.unique().tolist()
    if len(labels) - 2 <= 0:
        # All transparent or all black
        colors = np.vstack(([0, 0, 0], [255, 255, 255]))
    else:
        colors = np.vstack(([0, 0, 0], [255, 255, 255], generate_random_colors(len(labels) - 2)))

    for i, label in enumerate(labels):
        mask = label_tensor == label
        mask = mask.expand_as(recolorized).to(device)
        color = torch.tensor(colors[i]).view(3, 1, 1).to(device)
        if label == 0 and not colorize_line:
            continue
        recolorized = recolorized.to(device) * (~mask) + color.to(device) * mask
    return recolorized / 255.0


def recolorize_img(img):
    # img is the colorized image in [H,W,3].

    unique_colors = np.unique(img.reshape(-1, 3), axis=0)

    # Generate a new list of random colors excluding [255, 255, 255]
    non_white_colors = [color for color in generate_random_colors(len(unique_colors) - 1) if not np.array_equal(color, [255, 255, 255])]

    # Add [255, 255, 255] color back to the list
    if len(non_white_colors) == 0:
        new_colors = np.array([[255, 255, 255]])
        unique_colors = np.array([[255, 255, 255]])
    else:
        new_colors = np.vstack((non_white_colors, [255, 255, 255]))
        unique_colors = np.vstack(([color for color in unique_colors if not np.array_equal(color, [255, 255, 255])], [255, 255, 255]))
    # Create a dictionary mapping old colors to new colors
    color_mapping = dict(zip(map(tuple, unique_colors), new_colors))

    # Recolorize the image using the generated color mapping
    recolored_img = np.zeros_like(img, dtype=np.uint8)
    # Iterate over unique colors and their masks, change color using mask
    for color, new_color in color_mapping.items():
        mask = np.all(img == color, axis=-1)
        recolored_img[mask] = new_color
    return recolored_img


def recolorize_gt(gt):
    # Change all the black regions to the white one
    mask = (gt == [0, 0, 0]).all(axis=-1)
    gt[mask] = [255, 255, 255]
    gt = recolorize_img(gt)
    return torch.from_numpy(gt).permute(2, 0, 1).float() / 255.0


def transfer_alpha_channel(img_path, alpha_img_path, save_path):
    img1 = io.imread(img_path)  
    img2 = io.imread(alpha_img_path)  
    if img1.shape[:2] != img2.shape[:2]:
        raise ValueError("Image size mismatch.")
    alpha_channel = img2[:, :, 3]
    result_img = np.copy(img1)
    result_img[:, :, 3] = alpha_channel

    zero_alpha_pixels = alpha_channel == 0
    result_img[zero_alpha_pixels] = [255, 255, 255, 0]
    io.imsave(save_path, result_img)


def find_adjacent_labels(label_image, sq_neighbor=False):
    # Step 1: Label the connected components in the label image

    # Step 2: Create an adjacency dictionary
    adjacency_dict = {str(label): set() for label in np.unique(label_image) if label != 0}

    # Step 3: Define a square neighborhood for dilation
    neighborhood = square(3)

    # Step 4: Iterate through each label and find adjacent labels
    for current_label in adjacency_dict.keys():
        # Create a binary mask for the current label
        current_mask = label_image == int(current_label)
        if sq_neighbor:
            expanded_mask = binary_dilation(current_mask, footprint=neighborhood)
        else:
            expanded_mask = binary_dilation(current_mask)
        # Label the expanded regions and find unique labels
        expanded_labels = (expanded_mask & ~current_mask) * label_image
        # expanded_labels = expanded_labels[expanded_labels != int(current_label)]
        expanded_labels = expanded_labels[expanded_labels != 0]
        # Add adjacent labels to the adjacency dictionary
        adjacency_dict[current_label].update(np.unique(expanded_labels).tolist())
        adjacency_dict[current_label] = list(adjacency_dict[current_label])
    return adjacency_dict


def expand_label_img(label_image, num_iter=2):
    # Expand the label to fill the zero region (black line)
    neighborhood = square(3)
    labels = np.unique(label_image)
    for i in range(num_iter):
        for label in labels[1:]:  # Skip label 0 (background)
            region = label_image == label
            region_dilated = binary_dilation(region, footprint=neighborhood) * (label_image == 0)
            label_image[region_dilated] = label
    return label_image


def colorize_label_image(label_img_path, json_path, save_path, using="color"):
    # Load label img from the label_img_path and color json from the json path.
    # Output colorized img will be output to the save_path
    label_img_png = io.imread(label_img_path)
    label_img = labelpng_2_np(label_img_png)
    with open(json_path, "r") as f:
        json_dict = json.load(f)
    assert using in ["color", "label"]
    if using == "color":
        color_dict = json_dict
    if using == "label":
        color_dict = {k: default_colorbook[v] + [255] for k, v in json_dict.items()}
    color_index = np.array(list(color_dict.values()))
    color_index = np.insert(color_index, 0, [0, 0, 0, 255], axis=0)

    h, w = label_img.shape
    # colored_img = np.zeros((h, w, 4), dtype=np.uint8)
    colored_img = np.take(color_index, label_img[:, :], axis=0)
    colored_img = colored_img.astype(np.uint8)
    io.imsave(save_path, colored_img, check_contrast=False)
    return colored_img


def colorize_label_image_folder(label_dir, color_dir, save_dir):
    # Folder version of colorize_label_image(label_img_path,color_path,save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    label_paths = sorted(glob(os.path.join(label_dir, "*.png")))
    color_paths = sorted(glob(os.path.join(color_dir, "*.json")))
    if len(label_paths) == len(color_paths):
        num = len(label_paths)
        print(num, "Images are Loaded.")
        for i in tqdm(range(num)):
            save_path = os.path.join(save_dir, os.path.basename(label_paths[i]))
            colorize_label_image(label_paths[i], color_paths[i], save_path)
    elif len(label_paths) - 1 == len(color_paths):
        num = len(label_paths)
        print(num, "Images are Loaded.")
        for i in tqdm(range(num - 1)):
            save_path = os.path.join(save_dir, os.path.basename(label_paths[i + 1]))
            colorize_label_image(label_paths[i + 1], color_paths[i], save_path)
    else:
        print("label num is", len(label_paths), "and color dict num is", len(color_paths))
        assert False, "The number of label images and color jsons not match."


def eval_json_stage2(input_json_path, gt_json_path, gt_label_path, threshold=10):
    # compare two color json and calculate the accuracy
    input_color_dict = load_json(input_json_path)
    gt_color_dict = load_json(gt_json_path)
    assert len(input_color_dict.keys()) == len(gt_color_dict.keys()), "Input and output list length should match! " + input_json_path

    # input_label_img=labelpng_2_np(input_json_path.replace('.json','.png'))
    label_img = labelpng_2_np(io.imread(gt_label_path))
    index_counts = np.bincount(label_img.flatten()).tolist()
    acc_sum = 0
    pix_acc_sum = 0
    pix_acc_sum_wobg = 0
    acc_thres_sum = 0
    bmiou_sum = 0
    pix_bmiou_sum = 0

    acc_max = len(input_color_dict.keys())
    pix_acc_max = sum(index_counts[1:])  # Black lines are not needed to be counted!
    pix_acc_max_wobg = 0
    acc_thres_max = sum(np.array(index_counts) > threshold) - 1
    bmiou_max = 0
    pix_bmiou_max = 0

    background_color = [[0, 0, 0, 0], [255, 255, 255, 0]]

    for index in input_color_dict.keys():
        if gt_color_dict[index] not in background_color:
            pix_acc_max_wobg += index_counts[int(index)]
            if input_color_dict[index] == gt_color_dict[index]:
                pix_acc_sum_wobg += index_counts[int(index)]
        if input_color_dict[index] == gt_color_dict[index]:
            acc_sum += 1
            pix_acc_sum += index_counts[int(index)]
            if index_counts[int(index)] > threshold:
                acc_thres_sum += 1
        if input_color_dict[index] in background_color or gt_color_dict[index] in background_color:
            bmiou_max += 1
            pix_bmiou_max += index_counts[int(index)]
        if input_color_dict[index] in background_color and gt_color_dict[index] in background_color:
            bmiou_sum += 1
            pix_bmiou_sum += index_counts[int(index)]
    return {
        "acc": acc_sum / acc_max,
        "acc_thres": acc_thres_sum / acc_thres_max,
        "pix_acc": pix_acc_sum / pix_acc_max,
        "pix_acc_wobg": pix_acc_sum_wobg / pix_acc_max_wobg,
        "bmiou": bmiou_sum / bmiou_max if bmiou_max != 0 else 1.0,
        "pix_bmiou": pix_bmiou_sum / pix_bmiou_max if pix_bmiou_max != 0 else 1.0,
    }


def eval_json_stage1(input_json_path, gt_json_path, gt_label_path, threshold=10):
    # compare two color json and calculate the accuracy

    colorbook_path = "/".join(gt_json_path.split("/")[:-2] + ["colorbook.yml"])
    colorbook = ColorBook(colorbook_path)

    input_label_dict = load_json(input_json_path)
    gt_color_dict = load_json(gt_json_path)
    assert len(input_label_dict.keys()) == len(gt_color_dict.keys()), "Input and output list length should match!"

    # input_label_img=labelpng_2_np(input_json_path.replace('.json','.png'))
    label_img = labelpng_2_np(io.imread(gt_label_path))
    index_counts = np.bincount(label_img.flatten()).tolist()
    acc_sum = 0
    acc_thres_sum = 0
    pix_acc_sum = 0
    pix_acc_sum_foreground = 0

    acc_max = len(input_label_dict.keys())
    acc_thres_max = sum(np.array(index_counts[1:]) > threshold)
    pix_acc_max = sum(index_counts[1:])
    pix_acc_max_foreground = sum(index_counts[2:])

    for index in input_label_dict.keys():
        pix_count = index_counts[int(index)]

        gt_label = colorbook.get_color_name(gt_color_dict[index]).split(" ")[-1]
        if input_label_dict[index] == gt_label:
            acc_sum += 1
            if pix_count > threshold:
                acc_thres_sum += 1
            pix_acc_sum += pix_count
            if int(index) > 1:
                pix_acc_sum_foreground += pix_count

    return {
        "acc": acc_sum / acc_max,
        "acc_thres": acc_thres_sum / acc_thres_max,
        "pix_acc": pix_acc_sum / pix_acc_max,
        "pix_acc_wobg": pix_acc_sum_foreground / pix_acc_max_foreground,
    }



def eval_json_folder_orig(input_folder_path, gt_folder_path, result_folder_name, threshold=10):
    # Compare the evaluation results for two folders
    # Return three params:
    # Fisrt: an average dict with 'acc','acc_thres','pix_acc','bmiou','pix_bmiou'
    # Second: an average dict with different folder (such as 'michelle') 'acc','acc_thres','pix_acc','bmiou','pix_bmiou','count'
    # Third: 'acc','acc_thres','pix_acc','bmiou','pix_bmiou' for every image in each folder

    metrics = ["acc", "acc_thres", "pix_acc", "pix_acc_wobg", "bmiou", "pix_bmiou"]

    paired_json_list = []
    accumulated_values = {k: 0 for k in metrics}
    result_dict = {}
    all_results_dict = {}
    count = 0
    for item in os.listdir(input_folder_path):
        input_folder = os.path.join(input_folder_path, item, result_folder_name)
        gt_folder = os.path.join(gt_folder_path, item, "seg")
        label_folder = gt_folder
        json_list = sorted(glob(os.path.join(input_folder, "*.json")))
        for json_path in json_list:
            basename = os.path.basename(json_path)
            gt_json_path = os.path.join(gt_folder, basename)
            gt_label_path = os.path.join(label_folder, basename.replace("json", "png"))
            paired_json_list.append([json_path, gt_json_path, item, gt_label_path])
    for json_pair in tqdm(paired_json_list):
        acc_result = eval_json_stage2(json_pair[0], json_pair[1], json_pair[3], threshold)
        if json_pair[2] not in result_dict:
            result_dict[json_pair[2]] = {k: 0 for k in metrics + ["count"]}
            all_results_dict[json_pair[2]] = {k: [] for k in metrics}
        for key in accumulated_values:
            accumulated_values[key] += acc_result[key]
            result_dict[json_pair[2]][key] += acc_result[key]
            all_results_dict[json_pair[2]][key].append(acc_result[key])
        count += 1
        result_dict[json_pair[2]]["count"] += 1
    for file in result_dict:
        for key in metrics:
            result_dict[file][key] /= result_dict[file]["count"]
    average_values = {key: value / count for key, value in accumulated_values.items()}
    return average_values, result_dict, all_results_dict


def eval_json_folder(
    input_folder_path,
    gt_folder_path,
    threshold=10,
    stage=None,
    use_saved_seg=False,
    json_folder_name="seg",
):
    # Compare the evaluation results for two folders
    # Return three params:
    # Fisrt: an average dict with 'acc','acc_thres','pix_acc','bmiou','pix_bmiou'
    # Second: an average dict with different folder (such as 'michelle') 'acc','acc_thres','pix_acc','bmiou','pix_bmiou','count'
    # Third: 'acc','acc_thres','pix_acc','bmiou','pix_bmiou' for every image in each folder
    if stage is None:
        stage = "stage2"
    assert stage in ["stage1", "stage2", "end2end"]
    if stage == "stage1":
        metrics = ["acc", "acc_thres", "pix_acc", "pix_acc_wobg"]
    elif stage == "stage2":
        metrics = ["acc", "acc_thres", "pix_acc", "pix_acc_wobg", "bmiou", "pix_bmiou"]
    elif stage == "end2end":
        metrics = ["acc_s1", "acc_thres_s1", "pix_acc_s1", "pix_acc_wobg_s1", "acc", "acc_thres", "pix_acc", "pix_acc_wobg", "bmiou", "pix_bmiou"]

    # assert split in ["train", "test", "debug"]
    # if split == "train":
    #     input_folder_list = ["Aj", "Clarie", "Kaya", "Ortiz", "Kita", "TheBoss", "Doozy", "Remy", "Abe", "BigVegas", "Ryo", "Jackie"]
    # elif split == "test":
    #     input_folder_list = ["mousey", "Ichika", "michelle", "timmy", "Racer", "Mremireh_O_Desbiens", "Ty", "Sporty_Granny", "amy", "Bocchi"]
    # elif split == "debug":
    #     input_folder_list = ["Ichika", "Bocchi"]

    input_folder_list = os.listdir(gt_folder_path)

    # input_folder_list=os.listdir(input_folder_path)
    paired_json_list = []
    accumulated_values = {k: 0 for k in metrics}
    result_dict = {}
    all_results_dict = {}
    count = 0
    for item in input_folder_list:
        input_folder = os.path.join(input_folder_path, item)
        gt_folder = os.path.join(gt_folder_path, item, json_folder_name)
        label_folder = os.path.join(gt_folder_path, item, "seg")

        stage_folder_name = "stage1" if stage == "stage1" else "stage2"
        json_list = sorted(glob(os.path.join(input_folder, stage_folder_name, "*.json")))
        for json_path in json_list:
            basename = os.path.basename(json_path)
            gt_json_path = os.path.join(gt_folder, basename)
            gt_label_path = os.path.join(label_folder, basename.replace("json", "png"))
            if use_saved_seg:
                gt_label_path = json_path[:-5] + "_seg.png"
            paired_json_list.append([json_path, gt_json_path, item, gt_label_path])
    for json_pair in paired_json_list:
        json_path_stage1 = "stage1".join(json_pair[0].rsplit("stage2", 1))
        if stage == "stage1":
            acc_result = eval_json_stage1(json_path_stage1, json_pair[1], json_pair[3], threshold)
        elif stage == "stage2":
            acc_result = eval_json_stage2(json_pair[0], json_pair[1], json_pair[3], threshold)
        elif stage == "end2end":
            acc_result_s1 = eval_json_stage1(json_path_stage1, json_pair[1], json_pair[3], threshold)
            acc_result_s2 = eval_json_stage2(json_pair[0], json_pair[1], json_pair[3], threshold)
            acc_result = {**{k + "_s1": v for k, v in acc_result_s1.items()}, **acc_result_s2}
        char = json_pair[2]
        if char not in result_dict:
            result_dict[char] = {k: 0 for k in metrics + ["count"]}
            all_results_dict[char] = {k: [] for k in metrics}
        for key in accumulated_values:
            accumulated_values[key] += acc_result[key]
            result_dict[char][key] += acc_result[key]
            all_results_dict[char][key].append(acc_result[key])
        count += 1
        result_dict[char]["count"] += 1
    for file in result_dict:
        for key in metrics:
            result_dict[file][key] /= result_dict[file]["count"]
    average_values = {key: value / count for key, value in accumulated_values.items()}
    return average_values, result_dict, all_results_dict


def calculate_avg(data_list, split_interval=None, interval_dict={}, skip_first=True):
    # for a data_list, return the result which ignores the first elements for each split
    avg_dict = {}
    if split_interval == None:
        # Means all the folder will be evaluated as a sequence. No split is in the folder.
        if skip_first:
            avg_dict["avg"] = sum(data_list[1:]) / (len(data_list) - 1)
        else:
            avg_dict["avg"] = sum(data_list) / len(data_list)
    elif len(interval_dict) == 0:
        # No multiple camera poses, but multiple sequences.
        avg_dict["avg"] = sum(
            sum(group[1:split_interval]) / (split_interval - 1) if skip_first else sum(group[:split_interval]) / split_interval
            for group in (data_list[i : i + split_interval] for i in range(0, len(data_list), split_interval))
        ) / (len(data_list) // 20)
    else:
        start_mark = 0
        sum_avg = 0
        for key in interval_dict:
            end_mark = start_mark + interval_dict[key]
            sub_data_list = data_list[start_mark:end_mark]
            avg_dict[key] = sum(
                sum(group[1:split_interval]) / (split_interval - 1) if skip_first else sum(group[:split_interval]) / split_interval
                for group in (sub_data_list[i : i + split_interval] for i in range(0, len(sub_data_list), split_interval))
            ) / (len(sub_data_list) // 20)
            start_mark += interval_dict[key]
            sum_avg += avg_dict[key]
        avg_dict["avg"] = sum_avg / len(interval_dict)
    return avg_dict


def evaluate(result_tuple, mode="Default", split_interval=None, save_path=None, skip_first=True, stage=None):
    """
    Add a evaluation script which may ignore the first image (gt) in each interval.
    Thus, this evaluation script is mainly for the next-frame prediction.
    We have four mode:
    PaintBucket_Char:
        Return a dict with total 'acc','acc_thres','pix_acc','bmiou','pix_bmiou'
               a folder dict 'michelle','ammy',..... average
               and 'Longshot','Faceshot','Closeup' average
    PaintBucket_Object:
        Return a dict with total 'acc','acc_thres','pix_acc','bmiou','pix_bmiou'
               a folder dict 'camcorder','earphone',..... average
               and camera dict 'Longshot','Closeup' average
    PaintBueckt_MG / Default:
        Return a dict with total 'acc','acc_thres','pix_acc','bmiou','pix_bmiou'
               a folder dict 'Kandinsky1','Kandinsky2',..... average
    """
    assert stage in ["stage1", "stage2", "end2end"]
    if stage == "stage1":
        metrics = ["acc", "acc_thres", "pix_acc", "pix_acc_wobg"]
    elif stage == "stage2":
        metrics = ["acc", "acc_thres", "pix_acc", "pix_acc_wobg", "bmiou", "pix_bmiou"]
    elif stage == "end2end":
        metrics = ["acc_s1", "acc_thres_s1", "pix_acc_s1", "pix_acc_wobg_s1", "acc", "acc_thres", "pix_acc", "pix_acc_wobg", "bmiou", "pix_bmiou"]

    average_values, _, all_results_dict = result_tuple

    if mode == "PaintBucket_Char" or mode == "PaintBucket_Char_test":
        split_interval = 20 if split_interval is None else None
        interval_dict = {"Longshot": 100, "Faceshot": 100, "Closeup": 100}
    elif mode == "PaintBucket_Object":
        split_interval = 20 if split_interval is None else None
        interval_dict = {"Longshot": 40, "Closeup": 40}
    elif mode == "Default" or mode == "PaintBucket_MG" or mode == "PaintBucket_Real" or mode == "PaintBucket_Char_val" or mode == "PaintBucket_Char_debug":
        split_interval = None
        interval_dict = {}
    else:
        assert False, "Error! Evaluation mode not supported."
    output_avg_dict = {k: 0 for k in metrics}
    # like {'Longshot':{'acc':0,....}}
    if len(interval_dict) != 0:
        output_camera_dict = {key: {k: 0 for k in metrics} for key in interval_dict}
    # like {'michelle':{'acc':0,..}}
    output_datafolder_dict = {key: {k: 0 for k in metrics} for key in all_results_dict}
    for folder_name in all_results_dict:
        for key in average_values:
            data_list = all_results_dict[folder_name][key]
            output_dict = calculate_avg(data_list, split_interval, interval_dict, skip_first)
            for key_camera in output_dict:
                if key_camera == "avg":
                    output_avg_dict[key] += output_dict["avg"]
                    output_datafolder_dict[folder_name][key] += output_dict["avg"]
                else:
                    output_camera_dict[key_camera][key] += output_dict[key_camera]
    # Calculate the average result
    # Noted, for the folder with different number of images, the average accuracy is by folder not by image
    output_avg_dict = {key: value / len(all_results_dict) for key, value in output_avg_dict.items()}
    if len(interval_dict) != 0:
        output_camera_dict = {
            outer_key: {inner_key: inner_value / len(all_results_dict) for inner_key, inner_value in inner_dict.items()}
            for outer_key, inner_dict in output_camera_dict.items()
        }
    else:
        output_camera_dict = {}
    if save_path is not None:
        with open(save_path, "w", newline="") as csvfile:
            fieldnames = ["Scenario"] + metrics
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            row = {"Scenario": "AVG"}
            row.update(output_avg_dict)
            writer.writerow(row)
            writer.writerow({})  # Add a blank row

            for scenario, metrics in output_camera_dict.items():
                row = {"Scenario": scenario}
                row.update(metrics)
                writer.writerow(row)
            writer.writerow({})  # Add a blank row
            for scenario, metrics in output_datafolder_dict.items():
                row = {"Scenario": scenario}
                row.update(metrics)
                writer.writerow(row)

            print(f"CSV file saved at {save_path}")
    return output_avg_dict, output_camera_dict, output_datafolder_dict

def read_line_2_np(img_path, channel=4):
    img = Image.open(img_path)
    img_np = np.array(img)

    if img.mode == "RGBA":
        alpha_channel = img_np[:, :, 3]
        mask = alpha_channel > 100  # Line detection based on alpha value, default is 10
    elif img.mode == "RGB":
        grayscale = np.mean(img_np[:, :, :3], axis=2)
        mask = grayscale < 150  # Line detection based on grayscale value, default is 245

    line = np.zeros((*img_np.shape[:2], 4), dtype=np.uint8)
    line[:, :, :3] = 255  # Set all RGB to white
    line[:, :, 3] = np.where(mask, 255, 0)  # Set alpha: 255 for lines, 0 for background

    # Copy original RGB values to new image where there are lines
    line[mask, :3] = img_np[mask, :3]

    return line[..., :channel]

def process_gt(gt, seg):
    recolored_gt = np.zeros_like(gt)

    for seg_id in np.unique(seg):
        indices = np.where(seg == seg_id)

        colors = gt[indices[0], indices[1], :]
        unique_colors, counts = np.unique(colors, axis=0, return_counts=True)
        most_common_color = unique_colors[np.argmax(counts)]

        recolored_gt[indices[0], indices[1], :] = most_common_color

    return recolored_gt

def extract_black_line(color_line, thres=100):
    # Input: a color line with alpha channel.
    # Output: the black line extracted from the color line.
    processed_img = np.max(color_line[..., :3], axis=2)

    # Create a mask where processed_img > thres
    mask = processed_img > thres

    # Modify the RGBA values in the mask area
    color_line[mask] = [255, 255, 255, 0]
    color_line[...,:3][1-mask] = [0, 0, 0]
    return color_line

def merge_color_line(line_path,colorized_img_path,save_path):
    # merge the black part of the color line in the line_path with the colorized img in colorized_img_path, then save it.
    colorized_img = io.imread(colorized_img_path)
    line_img = io.imread(line_path)
    line_img = extract_black_line(line_img)

    line_alpha = line_img[..., 3] / 255.0
    line_rgb = line_img[...,:3]
    
    # Convert the colorized image to RGBA if it's not already
    if colorized_img.shape[2] == 3:
        colorized_img = np.dstack([colorized_img, np.ones(colorized_img.shape[:2], dtype=np.uint8) * 255])
    
    # Identify the [0,0,0,255] regions in the colorized image
    mask = np.all(colorized_img == [0, 0, 0, 255], axis=-1)
    
    # Inpaint the colorized image
    inpainted_image = 255*inpaint.inpaint_biharmonic(colorized_img[..., :3]/255, mask, channel_axis=-1)
    inpainted_image_alpha = np.dstack([inpainted_image, colorized_img[..., 3]])
    
    # Extract the black line from the line image
    blended_rgb = line_rgb * line_alpha[..., np.newaxis] + inpainted_image * (1 - line_alpha[..., np.newaxis])
    final_alpha = np.clip(line_alpha + inpainted_image_alpha[..., 3] * (1 - line_alpha), 0, 1)

    inpainted_image_alpha = np.dstack([blended_rgb, final_alpha * 255])
    
    # Save the final image
    io.imsave(save_path, inpainted_image_alpha.astype(np.uint8), check_contrast=False)