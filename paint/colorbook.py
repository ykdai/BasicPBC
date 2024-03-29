import copy
import numpy as np
import yaml
from skimage import io

from paint.color_redistribution import redistribute_colors


class ColorBook:
    def __init__(self, filename, ignore_colorline=True, book_img_param=None, name_list=None):
        if filename.endswith(".yml") or filename.endswith(".yaml"):
            with open(filename, "r") as f:
                color_list_dict = yaml.load(f, Loader=yaml.FullLoader)
            # self.color_list = []
            self.color_dict = {}
            for name, color in color_list_dict.items():
                if len(color) != 4:
                    raise ValueError(f"Invalid color format for {name} in {filename}.")
                colors = []
                for c in color:
                    if c is not None:
                        c = [int(i) for i in c[0].split(" ")]
                        if len(c) != 3:
                            raise ValueError(f"Invalid color format for {name} in {filename}.")
                    colors.append(c)
                # self.color_list.append(Color(name, colors))
                self.color_dict[name] = colors
        elif filename.endswith(".png"):
            # We do not support jpeg, since it may change the color.
            self.color_dict = self.load_from_colorbook_img(filename, book_img_param, name_list)
        else:
            assert False, "This type of file not supported."

        self.normal_list = []
        self.normal_name_list = []
        self.all_color_list = []
        self.all_color_name_list = []
        self.all_color_highlight_list = []  # 0 means higlight, 1 means normal, 2 means shadow
        self.color_map_dict = {}  # such as (255,255,255) -> 'face'

        for name, color in self.color_dict.items():
            name_split = name.split("_")
            if ignore_colorline and len(name_split) > 1 and name_split[1] in ["green", "blue", "red"]:
                pass
            else:
                self.normal_list.append(color[1])
                self.normal_name_list.append(name)
            self.all_color_list.extend(color)
            for i in range(4):
                if color[i] is not None:
                    tuple_color = tuple(color[i])
                    if tuple_color in self.color_map_dict:
                        print(tuple_color, "is repeated!")
                        assert False, "Please avoid repeated color!"
                    self.color_map_dict[tuple_color] = name
                    self.all_color_name_list.append(name)
                    self.all_color_highlight_list.append(i)
        self.normal_list = self.remove_duplicate_arrays(self.normal_list)
        self.all_color_list = self.remove_duplicate_arrays(self.all_color_list)
        self.normal_np = np.array(self.normal_list)
        self.all_color_np = np.array(self.all_color_list)

    def load_from_colorbook_img(self, filename, book_img_param, name_list):
        # This function in prepared from using Wechat screenshot for PaintMan ccf color
        colorbook_img = io.imread(filename)[:, :, :3]
        color_len = len(name_list)
        color_dict = {}
        if book_img_param == None:
            book_img_param = {"delta_h": 24, "delta_w": 44, "h": 62, "w": 30}
        for i in range(color_len):
            if i >= 16:
                h = book_img_param["h"] + (i - 16) * book_img_param["delta_h"]
            else:
                h = book_img_param["h"] + i * book_img_param["delta_h"]
            row_colors = []

            for j in range(4):
                if i >= 16:
                    w = book_img_param["w"] + (j + 4) * book_img_param["delta_w"]
                else:
                    w = book_img_param["w"] + j * book_img_param["delta_w"]
                # Get the color of the current pixel
                color = colorbook_img[h, w]
                # If the color is (255, 255, 255), set it to None
                if (color == np.array([255, 255, 255])).all():
                    row_colors.append(None)
                else:
                    row_colors.append(list(color))
            # Append the row's colors to the main list
            color_dict[name_list[i]] = row_colors
        print(color_dict)
        return color_dict

    def get_color(self, name):
        color_list = self.color_dict[name]
        return [x for x in color_list if x is not None]

    def get_color_name(self, color) -> str:
        color = color[:3]
        if color == [0, 0, 0]:
            return "background"
        for k, v in self.color_dict.items():
            if color in v:
                return k
        # TODO
        return "background"

    def remove_duplicate_arrays(self, array_list):
        new_array_list = []
        seen = set()
        for arr in array_list:
            if arr is not None:
                arr_tuple = tuple(arr)
                if arr_tuple not in seen:
                    new_array_list.append(arr)
                    seen.add(arr_tuple)
        return new_array_list

    def save_colorbook(self, filename, load_dict=None):
        save_dict = {}
        save_str = ""
        if load_dict is None:
            load_dict = self.color_dict
        for k, v in load_dict.items():
            if isinstance(v, list):
                for i in range(len(v)):
                    if v[i] is None:
                        v[i] = "null"
                    else:
                        v[i] = "[" + " ".join(str(x) for x in np.array(v[i])) + "]"
            save_dict[k] = "[" + ",".join(str(x) for x in v) + "]"
            save_str = save_str + k + str(": ") + save_dict[k] + "\n"
        # Save dict as yaml file
        with open(filename, "w") as f:
            f.write(save_str)

    def find_nearest_color(self, vec, color_type="normal", return_idx=False):
        if color_type == "normal":
            vectors = self.normal_np
        elif color_type == "all":
            vectors = self.all_color_np
        vec = np.array(vec)
        distances = np.linalg.norm(vectors - vec, axis=1)
        closest_vec_idx = np.argmin(distances)
        closest_vec = vectors[closest_vec_idx]
        if return_idx:
            return closest_vec, closest_vec_idx
        else:
            return closest_vec

    def generate_random_colors(self, num_colors, shuffle=True):
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

    def generate_random_colorbook_old(self, filename, forbidden_words=[]):
        # Generate a new colorbook with randomly picked color.
        # This function applies the most simple method, and re-distribute all colors in the color dict randomly.
        new_color_dict = copy.deepcopy(self.color_dict)
        new_color_np = self.generate_random_colors(len(self.all_color_list))
        index = 0
        for name, color in new_color_dict.items():
            for i in range(4):
                if color[i] is not None:
                    if name not in forbidden_words:
                        new_color_dict[name][i] = list(new_color_np[index])
                    index += 1
        self.save_colorbook(filename, new_color_dict)

    def generate_random_colorbook(self, filename, forbidden_words=[], random_color=False):
        # Generate a new colorbook with randomly picked color. Color will be reditributed with redistribute_colors functions.
        # This function simulates a N body problem as the basic color redistribution method.
        new_color_dict = copy.deepcopy(self.color_dict)
        new_normal_np = copy.deepcopy(self.normal_np)
        new_all_color_np = copy.deepcopy(self.all_color_np)
        # Change the normal list first.
        normal_list = [tuple(x) for x in self.normal_list]

        fixed_np = np.zeros_like(self.normal_np[:, 0])
        for i in range(len(self.normal_np)):
            if self.color_map_dict[normal_list[i]] in forbidden_words:
                fixed_np[i] = 1
        # Then, change the all color list.
        print("fixed_np", fixed_np)
        new_normal_np = redistribute_colors(new_normal_np, fixed_np, random_color)
        print("new_normal:", new_normal_np)
        fixed_all_color_np = np.zeros_like(self.all_color_np[:, 0])
        for i in range(len(self.all_color_np)):
            if self.color_map_dict[tuple(self.all_color_list[i])] in forbidden_words:
                fixed_all_color_np[i] = 1
            elif tuple(self.all_color_np[i]) in normal_list:
                normal_index = normal_list.index(tuple(self.all_color_np[i]))
                fixed_all_color_np[i] = 1
                new_all_color_np[i] = new_normal_np[normal_index]
        new_all_color_np = redistribute_colors(new_all_color_np, fixed_all_color_np, random_color)
        index = 0
        for name, color in new_color_dict.items():
            for i in range(4):
                if color[i] is not None:
                    if name not in forbidden_words:
                        new_color_dict[name][i] = list(new_all_color_np[index])
                    index += 1
        self.save_colorbook(filename, new_color_dict)
