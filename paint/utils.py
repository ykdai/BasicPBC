import csv
import json
import numpy as np
import os
import torch
from glob import glob
from PIL import Image
from scipy import stats
from skimage import io,img_as_ubyte
from skimage.morphology import disk, binary_dilation
from skimage.restoration import inpaint
from tqdm import tqdm


def load_json(file_path):
    with open(file_path, "r") as f:
        file_dict = json.load(f)
    return file_dict


def dump_json(save_dict, save_path):
    file_json = json.dumps(save_dict)
    with open(save_path, "w") as f:
        f.write(file_json)


def np_2_labelpng(input_np, save_path=None):
    """
    Transfer a 1-ch label image (numpy format) into the 3-ch label png, it can cover the label with index less than 256^3
    Input is a numpy and output is the image. R*256^2+G^256+B is the label for each pixel.
    """
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
    """
    Transfer a 3-ch or 4-ch label image into the 1-ch label image (numpy format).
    Input is a image (skimage) and output is the np. R*256^2+G^256+B is the label for each pixel.
    For 4-ch label image transparent region is -1.
    """
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
    img = Image.open(img_path).convert("RGBA")
    img_np = np.array(img).astype(type)
    if len(img_np.shape) == 2:
        img_np = np.tile(img_np[..., None], (1, 1, channel))
    else:
        img_np = img_np[..., :channel]
    return img_np


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
    print(len(color_mapping.items()))
    for color, new_color in color_mapping.items():
        mask = np.all(img == color, axis=-1)
        recolored_img[mask] = new_color
    
    image_save = Image.fromarray(recolored_img)
    path = 'recolorized_image.png'
    image_save.save(path)

    return recolored_img


def recolorize_gt(gt):
    # Change all the black regions to the white one
    mask = (gt == [0, 0, 0]).all(axis=-1)
    gt[mask] = [255, 255, 255]
    gt = recolorize_img(gt)
    return torch.from_numpy(gt).permute(2, 0, 1).float() / 255.0


def colorize_label_image(label_img_path, json_path, save_path):
    # Load label img from the label_img_path and color json from the json path.
    # Output colorized img will be output to the save_path
    label_img_png = io.imread(label_img_path)
    label_img = labelpng_2_np(label_img_png)
    with open(json_path, "r") as f:
        json_dict = json.load(f)
    color_dict = json_dict
    color_index = np.array(list(color_dict.values()))
    line_color = [0, 0, 0, 255] if len(color_index[0]) == 4 else [0, 0, 0]
    color_index = np.insert(color_index, 0, line_color, axis=0)

    h, w = label_img.shape
    # colored_img = np.zeros((h, w, 4), dtype=np.uint8)
    colored_img = np.take(color_index, label_img[:, :], axis=0)
    colored_img = colored_img.astype(np.uint8)
    io.imsave(save_path, colored_img, check_contrast=False)
    return colored_img


def eval_json(input_json_path, gt_json_path, threshold=10):
    # compare two color json and calculate the accuracy
    input_color_dict = load_json(input_json_path)
    gt_color_dict = load_json(gt_json_path)
    assert len(input_color_dict.keys()) == len(gt_color_dict.keys()), "Input and output list length should match! " + input_json_path

    # input_label_img=labelpng_2_np(input_json_path.replace('.json','.png'))
    label_img = labelpng_2_np(io.imread(gt_json_path.replace(".json", ".png")))
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

    for index in input_color_dict.keys():
        if not gt_color_dict[index] == [255, 255, 255, 0]:
            pix_acc_max_wobg += index_counts[int(index)]
            if input_color_dict[index] == gt_color_dict[index]:
                pix_acc_sum_wobg += index_counts[int(index)]
        if input_color_dict[index] == gt_color_dict[index]:
            acc_sum += 1
            pix_acc_sum += index_counts[int(index)]
            if index_counts[int(index)] > threshold:
                acc_thres_sum += 1
        if input_color_dict[index] == [255, 255, 255, 0] or gt_color_dict[index] == [255, 255, 255, 0]:
            bmiou_max += 1
            pix_bmiou_max += index_counts[int(index)]
        if input_color_dict[index] == [255, 255, 255, 0] and gt_color_dict[index] == [255, 255, 255, 0]:
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


def eval_json_folder(input_folder_path, gt_folder_path, result_folder_name, threshold=10):
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
        acc_result = eval_json(json_pair[0], json_pair[1], threshold)
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


def calculate_avg(data_list, split_interval=None, interval_dict={}):
    # for a data_list, return the result which ignores the first elements for each split
    avg_dict = {}
    if split_interval == None:
        # Means all the folder will be evaluated as a sequence. No split is in the folder.
        avg_dict["avg"] = sum(data_list[1:]) / (len(data_list) - 1)
    elif len(interval_dict) == 0:
        # No multiple camera poses, but multiple sequences.
        avg_dict["avg"] = sum(
            sum(group[1:split_interval]) / (split_interval - 1)
            for group in (data_list[i : i + split_interval] for i in range(0, len(data_list), split_interval))
        ) / (len(data_list) // 20)
    else:
        start_mark = 0
        sum_avg = 0
        for key in interval_dict:
            end_mark = start_mark + interval_dict[key]
            sub_data_list = data_list[start_mark:end_mark]
            avg_dict[key] = sum(
                sum(group[1:split_interval]) / (split_interval - 1)
                for group in (sub_data_list[i : i + split_interval] for i in range(0, len(sub_data_list), split_interval))
            ) / (len(sub_data_list) // 20)
            start_mark += interval_dict[key]
            sum_avg += avg_dict[key]
        avg_dict["avg"] = sum_avg / len(interval_dict)
    return avg_dict


def evaluate(result_tuple, mode="Default", split_interval=None, save_path=None):
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
    metrics = ["acc", "acc_thres", "pix_acc", "pix_acc_wobg", "bmiou", "pix_bmiou"]

    average_values, _, all_results_dict = result_tuple

    if mode == "PaintBucket_Char":
        split_interval = 20 if split_interval is None else None
        interval_dict = {"Longshot": 100, "Faceshot": 100, "Closeup": 100}
    elif mode == "PaintBucket_Object":
        split_interval = 20 if split_interval is None else None
        interval_dict = {"Longshot": 40, "Closeup": 40}
    elif mode == "Default" or mode == "PaintBucket_MG" or mode == "PaintBucket_Real":
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
            output_dict = calculate_avg(data_list, split_interval, interval_dict)
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

if __name__ == "__main__":
    # Example file paths
    line_path = 'dataset/PaintBucket_demo/tangyuan_anime/line/0000.png'
    colorized_img_path = 'dataset/PaintBucket_demo/tangyuan_anime/tangyuan_anime/0000.png'
    save_path = 'merged_image.png'
    
    # Call the merge_line function
    merge_color_line(line_path, colorized_img_path, save_path)
    
    print(f'Merged image saved to {save_path}')