import os
import torch
from torch import nn as nn
import shutil

import numpy as np
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from skimage import io
from paint.utils import recolorize_img

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from paint.utils import eval_json_folder, evaluate, colorize_label_image, dump_json, load_json, np_2_labelpng

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

@MODEL_REGISTRY.register()
class PBCModel_ref(SRModel):

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]

        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)
            # load pretrained model
            load_path = self.opt["path"].get("pretrain_network_g", None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt["path"].get("strict_load_g", True), "params_ema")
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        self.l_ce = build_loss(train_opt["l_ce"]).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.data = data
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                self.data[key] = data[key].to(self.device)
            elif isinstance(data[key], list) and isinstance(data[key][0], torch.Tensor):
                self.data[key] = [x.to(self.device) for x in data[key]]


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.data)

        for k, v in self.data.items():
            self.data[k] = v[0]
        pred = {**self.data, **self.output}

        l_total = 0
        loss_dict = OrderedDict()
        # l1 loss

        if pred["skip_train"]:
            return 0

        loss = pred["loss"]  # / self.opt['datasets']['train']['batch_size_per_gpu']
        loss_dict["acc"] = torch.tensor(pred["accuracy"]).to(self.device)
        if "area_accuracy" in pred:
            loss_dict["area_acc"] = torch.tensor(pred["area_accuracy"]).to(self.device)
        if "valid_acc" in pred:
            loss_dict["valid_acc"] = torch.tensor(pred["valid_accuracy"]).to(self.device)

        loss_dict["loss_total"] = self.l_ce(loss)
        l_total += loss
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.data)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.data)

        if not hasattr(self, "net_g_ema"):
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt["rank"] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt["name"]
        gt_folder_path = dataloader.dataset.opt["root"]  # self.opt['datasets']['val']['root']
        with_metrics = self.opt["val"].get("metrics") is not None
        save_img = self.opt["val"].get("save_img", False)
        save_csv = self.opt["val"].get("save_csv", False)
        accu = self.opt["val"].get("accu", False)
        self_prop = self.opt["val"].get("self_prop", False)
        frame_per_clip = self.opt["val"].get("frame_per_clip", 10000)
        based_on_ref = self.opt["val"].get("based_on_ref", True)
        stage = self.opt["val"].get("stage", "stage2")

        assert stage in ["stage1", "stage2", "end2end"]

        json_color_folder_name = "json_color" if dataset_name in ["PaintBucket_Char", "PaintBucket_Char_val"] else "seg"
        use_label_index = dataset_name in ["PaintBucket_Char", "PaintBucket_Char_val"]

        if with_metrics:
            if not hasattr(self, "metric_results"):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt["val"]["metrics"].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}

        save_path = osp.join(self.opt["path"]["visualization"], str(current_iter), dataset_name)

        test_args = {
            "mode": stage,
            "ref": "ref" if based_on_ref else "prev",
            "save_path": save_path,
            "save_img": save_img,
            "is_val": True,
            "accu": accu,
            "json_color": json_color_folder_name,
            "index": "label" if use_label_index else "seg",
            "self_prop": self_prop,
            "frame_per_clip": frame_per_clip,
        }

        if hasattr(self, "net_g_ema"):
            model_inference = ModelInference(self.net_g_ema, dataloader, test_args)
        else:
            model_inference = ModelInference(self.net_g, dataloader, test_args)

        model_inference.inference()

        results = eval_json_folder(save_path, gt_folder_path, stage=stage, json_folder_name=json_color_folder_name)
        skip_first = False if based_on_ref else True
        if save_csv:
            csv_save_path = os.path.join(save_path, "metrics.csv")
            avg_dict, _, _ = evaluate(results, mode=dataset_name, save_path=csv_save_path, skip_first=skip_first, stage=stage)
        else:
            avg_dict, _, _ = evaluate(results, mode=dataset_name, skip_first=skip_first, stage=stage)

        for metric, value in avg_dict.items():
            self.metric_results[metric] = value

        if with_metrics:
            for metric in self.metric_results.keys():
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}"
            if hasattr(self, "best_metric_results"):
                log_str += (
                    f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ ' f'{self.best_metric_results[dataset_name][metric]["iter"]} iter'
                )
            log_str += "\n"

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f"metrics/{dataset_name}/{metric}", value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        # Just output the line for test
        out_dict["line"] = self.data["image1"].detach().cpu()
        return out_dict


@MODEL_REGISTRY.register()
class PBCModelStage1(SRModel):

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]

        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)
            # load pretrained model
            load_path = self.opt["path"].get("pretrain_network_g", None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt["path"].get("strict_load_g", True), "params_ema")
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        self.l_ce = build_loss(train_opt["l_ce"]).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.data = data
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                self.data[key] = data[key].to(self.device)
            elif isinstance(data[key], list) and isinstance(data[key][0], torch.Tensor):
                self.data[key] = [x.to(self.device) for x in data[key]]

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.data)

        for k, v in self.data.items():
            self.data[k] = v[0]
        pred = {**self.data, **self.output}

        l_total = 0
        loss_dict = OrderedDict()
        # l1 loss

        if pred["skip_train"]:
            return 0

        loss = pred["loss"]  # / self.opt['datasets']['train']['batch_size_per_gpu']
        loss_dict["acc"] = torch.tensor(pred["accuracy"]).to(self.device)

        loss_dict["loss_total"] = self.l_ce(loss)
        l_total += loss
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.data)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.data)

        if not hasattr(self, "net_g_ema"):
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt["rank"] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt["name"]
        gt_folder_path = dataloader.dataset.opt["root"]  # self.opt['datasets']['val']['root']
        with_metrics = self.opt["val"].get("metrics") is not None
        use_label = self.opt["val"].get("use_label", False)
        save_img = self.opt["val"].get("save_img", False)
        save_csv = self.opt["val"].get("save_csv", False)

        color_line = True if dataset_name == "PaintBucket_Real" else False
        json_folder_name = "seg" if dataset_name in ["PaintBucket_Char_test", "PaintBucket_Char_debug", "PaintBucket_Real"] else "json_color"

        if with_metrics:
            if not hasattr(self, "metric_results"):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt["val"]["metrics"].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if hasattr(self, "net_g_ema"):
            model_inference = ModelInference(self.net_g_ema, dataloader)
        else:
            model_inference = ModelInference(self.net_g, dataloader)

        self.net_g.train()
        save_path = osp.join(self.opt["path"]["visualization"], str(current_iter), dataset_name)
        model_inference.process_stage1(save_path, save_img=save_img, is_val=True)

        results = eval_json_folder(save_path, gt_folder_path, stage="stage1", json_folder_name=json_folder_name)
        csv_save_path = os.path.join(save_path, "metrics.csv") if save_csv else None
        avg_dict, _, _ = evaluate(results, mode=dataset_name, save_path=csv_save_path, skip_first=False, stage="stage1")

        for metric, value in avg_dict.items():
            self.metric_results[metric] = value

        if with_metrics:
            for metric in self.metric_results.keys():
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}"
            if hasattr(self, "best_metric_results"):
                log_str += (
                    f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ ' f'{self.best_metric_results[dataset_name][metric]["iter"]} iter'
                )
            log_str += "\n"

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f"metrics/{dataset_name}/{metric}", value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        # Just output the line for test
        out_dict["line"] = self.data["image1"].detach().cpu()
        return out_dict


class ModelInference:
    def __init__(self, model, test_loader, test_args: dict):
        self.model = model
        self.test_loader = test_loader
        self.test_args = test_args

        self.mode = test_args["mode"]
        self.ref = test_args["ref"]

    def dis_data_to_cuda(self, data):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda()
            elif isinstance(data[key], list) and isinstance(data[key][0], torch.Tensor):
                data[key] = [x.cuda() for x in data[key]]
        return data

    def inference(self):
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                self.inference_step(batch)
        self.model.train()

    def inference_step(self, batch):
        output = self.model(self.dis_data_to_cuda(batch))
        if self.mode in ["stage1", "end2end"]:
            self.process_stage1(batch, output)
        if self.mode in ["stage2", "end2end"]:
            if self.ref == "prev":
                self.process_stage2_frame_by_frame(batch, output)
            elif self.ref == "ref":
                self.process_stage2_based_on_reference(batch, output)

    def process_stage2_frame_by_frame(self, batch, pred):
        # Process the line arts frame by frame and save them at save_path
        # For example, if the save_path is 'aug_iter360k'
        # the output images will be saved at: '{img_load_path}/{glob_folder(like michelle)}/aug_iter360k/0001.png'
        save_path = self.test_args["save_path"]
        load_folder = "seg"
        save_img = self.test_args["save_img"]
        is_val = self.test_args["is_val"]
        accu = self.test_args["accu"]
        self_prop = self.test_args["self_prop"]
        frame_per_clip = self.test_args["frame_per_clip"]

        root, name_str = osp.split(batch["file_name"][0])
        prev_index = int(name_str) - 1
        prev_name_str = str(prev_index).zfill(len(name_str))
        prev_json_path = osp.join(osp.split(root)[0], load_folder, prev_name_str + ".json")
        if is_val:
            save_folder = osp.join(save_path, osp.split(osp.split(root)[0])[-1])
        else:
            save_folder = osp.join(osp.split(root)[0].replace("test_data", "results"), save_path)
        save_folder = osp.join(save_folder, "stage2")
        start_frames = [0] if "PaintBucket_Real" in root else [idx for idx in range(0, 10000, frame_per_clip)]
        if prev_index in start_frames:
            mkdir(save_folder)
            shutil.copy(prev_json_path, save_folder)  # Copy the 0000.json to the result folder
            if save_img or self_prop:
                label0_path = osp.join(osp.split(root)[0], "seg", prev_name_str + ".png")
                img0_save_path = osp.join(save_folder, prev_name_str + ".png")
                colorize_label_image(label0_path, prev_json_path, img0_save_path)
        if self_prop:
            prev_json_path = osp.join(save_folder, prev_name_str + ".json")
            color_dict = load_json(prev_json_path)
            prev_color_img = io.imread(osp.join(save_folder, prev_name_str + ".png"))[:, :, :3]
            mask = (prev_color_img == [0, 0, 0]).all(axis=-1)
            prev_color_img[mask] = [255, 255, 255]  # Change all the black regions to the white one
            prev_color_img = recolorize_img(prev_color_img)
            recolorized_img = torch.from_numpy(prev_color_img).permute(2, 0, 1).float() / 255.0
            batch["color1"] = recolorized_img.unsqueeze(0)
        else:
            color_dict = load_json(prev_json_path)
        # color_dict['0']=[0,0,0,255] #black line
        json_save_path = osp.join(save_folder, name_str + ".json")

        match = pred["matches0"].cpu().numpy()
        match_scores = pred["match_scores"].cpu().numpy()

        color_next_frame = {}
        if not accu:
            for i, item in enumerate(match):
                if item == -1:
                    # This segment cannot be matched
                    color_next_frame[str(i + 1)] = [0, 0, 0, 0]
                else:
                    color_next_frame[str(i + 1)] = color_dict[str(item + 1)]
        else:
            for i, scores in enumerate(match_scores):
                color_lookup = np.array([color_dict[str(i + 1)] if str(i + 1) in color_dict else [0, 0, 0, 0] for i in range(len(scores))])
                unique_colors = np.unique(color_lookup, axis=0)
                accumulated_probs = [np.sum(scores[np.all(color_lookup == color, axis=1)]) for color in unique_colors]
                color_next_frame[str(i + 1)] = unique_colors[np.argmax(accumulated_probs)].tolist()
        # color_next_frame.pop('0')
        dump_json(color_next_frame, json_save_path)
        if save_img or self_prop:
            label_path = osp.join(osp.split(root)[0], "seg", name_str + ".png")
            img_save_path = json_save_path.replace(".json", ".png")
            colorize_label_image(label_path, json_save_path, img_save_path)

    def process_stage2_based_on_reference(self, batch, pred):
        save_path = self.test_args["save_path"]
        save_img = self.test_args["save_img"]
        is_val = self.test_args["is_val"]
        accu = self.test_args["accu"]
        json_color_folder_name = self.test_args["json_color"]
        index = self.test_args["index"]

        root, name_str = osp.split(batch["file_name"][0])
        if "match_ref_idx" in pred:
            # ref_index = 0
            ref_index = pred["match_ref_idx"]
        else:
            ref_index = 0
        ref_name_str = str(ref_index).zfill(len(name_str))
        ref_json_color_path = osp.join(osp.split(root)[0], "ref", json_color_folder_name, ref_name_str + ".json")
        if is_val:
            save_folder = osp.join(save_path, osp.split(osp.split(root)[0])[-1])
        else:
            save_folder = osp.join(osp.split(root)[0].replace("test_data", "results"), save_path)
        save_folder = osp.join(save_folder, "stage2")
        mkdir(save_folder)
        color_dict = load_json(ref_json_color_path)
        # color_dict['0']=[0,0,0,255] #black line
        color_dict = {int(idx): color for idx, color in color_dict.items()}

        if index == "label":
            ref_json_index_path = ref_json_color_path.replace(json_color_folder_name, "json_index")
            index_dict = load_json(ref_json_index_path)
            index_dict = {int(idx): pair[-1] for idx, pair in index_dict.items()}
            index_dict = {label_idx: seg_idx for seg_idx, label_idx in index_dict.items()}

        json_save_path = osp.join(save_folder, name_str + ".json")

        match = pred["matches0"].cpu().numpy()
        match_scores = pred["match_scores"].cpu().numpy()

        color_next_frame = {}
        if not accu:
            for i, item in enumerate(match):
                idx = item + 1 if index == "seg" else index_dict.get(item + 1, 1)
                if item == -1:
                    # This segment cannot be matched
                    color_next_frame[str(i + 1)] = [0, 0, 0, 0]
                else:
                    color_next_frame[str(i + 1)] = color_dict[idx]
        else:
            for i, scores in enumerate(match_scores):
                # idx = i+1 if not use_label_index else index_dict.get(i+1, 1)
                if index == "seg":
                    color_lookup = np.array([color_dict[idx] if idx in color_dict else [0, 0, 0, 0] for idx in range(1, len(scores) + 1)])
                else:
                    color_lookup = np.array([color_dict[index_dict.get(label_idx, 1)] for label_idx in batch["seg_list_refs"][0].tolist() + [-1]])
                unique_colors = np.unique(color_lookup, axis=0)
                accumulated_probs = [np.sum(scores[np.all(color_lookup == color, axis=1)]) for color in unique_colors]
                color_next_frame[str(i + 1)] = unique_colors[np.argmax(accumulated_probs)].tolist()
        # color_next_frame.pop('0')
        dump_json(color_next_frame, json_save_path)
        if save_img:
            label_path = osp.join(osp.split(root)[0], "seg", name_str + ".png")
            img_save_path = json_save_path.replace(".json", ".png")
            colorize_label_image(label_path, json_save_path, img_save_path)

    def process_stage1(self, batch, pred):
        save_path = self.test_args["save_path"]
        save_img = self.test_args["save_img"]
        save_seg = False
        is_val = self.test_args["is_val"]

        root, name_str = osp.split(batch["file_name"][0])
        if is_val:
            save_folder = osp.join(save_path, osp.split(osp.split(root)[0])[-1])
        else:
            save_folder = osp.join(osp.split(root)[0].replace("test_data", "results"), save_path)
        save_folder = osp.join(save_folder, "stage1")
        mkdir(save_folder)
        json_save_path = osp.join(save_folder, name_str + ".json")

        # assert not output["skip_train"]
        labels = pred["pred_labels"]

        labels_all_seg = {}
        for i, label in enumerate(labels):
            labels_all_seg[str(i + 1)] = label
        dump_json(labels_all_seg, json_save_path)
        if save_img:
            img_save_path = json_save_path.replace(".json", ".png")
            if save_seg:
                label_img = np_2_labelpng(batch["segment"].squeeze().cpu().numpy())
                label_save_path = img_save_path[:-4] + "_seg.png"
                io.imsave(label_save_path, label_img, check_contrast=False)
            else:
                label_save_path = osp.join(osp.split(root)[0], "seg", name_str + ".png")
            colorize_label_image(label_save_path, json_save_path, img_save_path, using="label")

    def visualize_text_matching(self, save_path):
        # Process the line arts frame by frame and save them at save_path
        # For example, if the save_path is 'aug_iter360k'
        # the output images will be saved at: '{img_load_path}/{glob_folder(like michelle)}/aug_iter360k/0001.png'
        # DONE: Need to add a module to output images like 0000.png to ensure alignment for evaluation statistics
        with torch.no_grad():
            self.model.eval()
            for data in tqdm(self.test_loader):
                root, name_str = osp.split(data["file_name"][0])
                save_folder = osp.join(save_path, osp.split(osp.split(root)[0])[-1], name_str)
                mkdir(save_folder)
                match_tensor = self.model(self.dis_data_to_cuda(data))
                text_match_scores = match_tensor["text_scores"]  # b, n(number of segments), m(number of tags)
                tag_len = text_match_scores.shape[1]
                text_match_scores = nn.functional.gumbel_softmax(
                    text_match_scores, tau=1.0, dim=1, hard=False
                )  # nn.functional.softmax(text_match_scores, dim=1)
                text_list = list(set(t[0] for t in data["text1"]))
                for i, text in enumerate(text_list):
                    scores = text_match_scores[:, i].squeeze().cpu().numpy()
                    segment = data["segment0"].squeeze().cpu().numpy()
                    vis_img = np.zeros((*segment.shape, 3))
                    for j, score in enumerate(scores):
                        # color = np.array([255, 255 * (1-score**0.5), 255 * (1-score**0.5)])
                        color = np.array([255, 255, 255]) * (1 - score) + score * np.array([100, 100, 255])
                        vis_img[segment == j + 1] = color.astype(np.uint8)

                    # save vis_img
                    img_save_path = osp.join(save_folder, text + ".jpeg")
                    io.imsave(img_save_path, vis_img.astype(np.uint8), check_contrast=False)
