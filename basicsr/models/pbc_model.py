import numpy as np
import os
import os.path as osp
import random
import shutil
import torch
from collections import OrderedDict
from glob import glob
from skimage import io
from torch import nn as nn
from torch.nn import init as init
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger, set_random_seed
from basicsr.utils.registry import MODEL_REGISTRY
from paint.utils import colorize_label_image, dump_json, eval_json_folder, evaluate, load_json, read_img_2_np, recolorize_gt, merge_color_line


@MODEL_REGISTRY.register()
class PBCModel(SRModel):

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
        white_list = ["file_name"]
        for key in data.keys():
            if key not in white_list:
                self.data[key] = data[key].to(self.device)

    def optimize_parameters(self, current_iter):

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.data)

        for k, v in self.data.items():
            self.data[k] = v[0]
        pred = {**self.data, **self.output}

        if pred["skip_train"]:
            return

        l_total = 0
        loss_dict = OrderedDict()

        loss = pred["loss"]  # / self.opt['datasets']['train']['batch_size_per_gpu']
        loss_dict["acc"] = torch.tensor(pred["accuracy"]).to(self.device)
        loss_dict["area_acc"] = torch.tensor(pred["area_accuracy"]).to(self.device)
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
        gt_folder_path = dataloader.dataset.opt["root"]
        with_metrics = self.opt["val"].get("metrics") is not None
        save_img = self.opt["val"].get("save_img", False)
        save_csv = self.opt["val"].get("save_csv", False)

        if with_metrics:
            if not hasattr(self, "metric_results"):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt["val"]["metrics"].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if hasattr(self, "net_g_ema"):
            model_inference = ModelInference(self.net_g_ema, dataloader)
        else:
            model_inference = ModelInference(self.net_g, dataloader)

        self.net_g.train()
        save_path = osp.join(self.opt["path"]["visualization"], str(current_iter), dataset_name)
        model_inference.inference_frame_by_frame(save_path, save_img)
        results = eval_json_folder(save_path, gt_folder_path, "")
        if save_csv:
            csv_save_path = os.path.join(save_path, "metrics.csv")
            avg_dict, _, _ = evaluate(results, mode=dataset_name, save_path=csv_save_path)
        else:
            avg_dict, _, _ = evaluate(results, mode=dataset_name)

        self.metric_results["acc"] = avg_dict["acc"]
        self.metric_results["acc_thres"] = avg_dict["acc_thres"]
        self.metric_results["pix_acc"] = avg_dict["pix_acc"]
        self.metric_results["pix_acc_wobg"] = avg_dict["pix_acc_wobg"]
        self.metric_results["bmiou"] = avg_dict["bmiou"]
        self.metric_results["pix_bmiou"] = avg_dict["pix_bmiou"]

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
        out_dict["line"] = self.data["line_ref"].detach().cpu()
        """
        out_dict['result']= self.blend.detach().cpu()
        out_dict['flare']=self.flare_hat.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        """
        return out_dict


class ModelInference:
    def __init__(self, model, test_loader, seed=42):

        self._set_seed(seed)

        self.test_loader = test_loader
        self.model = model
        self.model.eval()  # Set the model to evaluation mode

    def __del__(self):
        self._recover_seed()

    def _set_seed(self, seed):
        self.py_rng_state0 = random.getstate()
        self.np_rng_state0 = np.random.get_state()
        self.torch_rng_state0 = torch.get_rng_state()
        set_random_seed(seed)

    def _recover_seed(self):
        if hasattr(self, 'py_rng_state0'):
            random.setstate(self.py_rng_state0)
        if hasattr(self, 'np_rng_state0'):
            np.random.set_state(self.np_rng_state0)
        if hasattr(self, 'torch_rng_state0'):
            torch.set_rng_state(self.torch_rng_state0)

    def dis_data_to_cuda(self, data):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda()
        return data

    def inference_frame_by_frame(self, save_path, save_img=False):
        # Process the line arts frame by frame and save them at save_path
        with torch.no_grad():
            self.model.eval()
            for test_data in tqdm(self.test_loader):
                line_root, name_str = osp.split(test_data["file_name"][0])
                character_root = osp.split(line_root)[0]
                prev_index = int(name_str) - 1
                prev_name_str = str(prev_index).zfill(len(name_str))
                ref_json_path = osp.join(character_root, "seg", prev_name_str + ".json")
                save_folder = osp.join(save_path, osp.split(character_root)[-1])

                if prev_index == 0:
                    os.makedirs(save_folder, exist_ok=True)
                    shutil.copy(ref_json_path, save_folder)
                    if save_img:
                        gt0_path = osp.join(character_root, "gt", prev_name_str + ".png")
                        shutil.copy(gt0_path, save_folder)

                color_dict = load_json(ref_json_path)
                json_save_path = osp.join(save_folder, name_str + ".json")

                match_tensor = self.model(self.dis_data_to_cuda(test_data))
                match_scores = match_tensor["match_scores"].cpu().numpy()

                color_next_frame = {}
                unmatch_color = [0] * len(list(color_dict.values())[0])
                for i, scores in enumerate(match_scores):
                    color_lookup = np.array([(color_dict[str(i + 1)] if str(i + 1) in color_dict else unmatch_color) for i in range(len(scores))])
                    unique_colors = np.unique(color_lookup, axis=0)
                    accumulated_probs = [np.sum(scores[np.all(color_lookup == color, axis=1)]) for color in unique_colors]
                    color_next_frame[str(i + 1)] = unique_colors[np.argmax(accumulated_probs)].tolist()
                dump_json(color_next_frame, json_save_path)

                if save_img:
                    label_path = osp.join(character_root, "seg", name_str + ".png")
                    img_save_path = json_save_path.replace(".json", ".png")
                    colorize_label_image(label_path, json_save_path, img_save_path)

    def inference_multi_gt(self, save_path, keep_line=False):
        with torch.no_grad():
            self.model.eval()
            characters = set()
            for test_data in tqdm(self.test_loader):
                line_root, name_str = osp.split(test_data["file_name"][0])
                character_root, _ = osp.split(line_root)
                _, character_name = osp.split(character_root)

                save_folder = osp.join(save_path, character_name)
                save_folder_keepline = osp.join(save_path, character_name+'_keepline')
                if character_name not in characters:
                    characters.add(character_name)
                    os.makedirs(save_folder, exist_ok=True)
                    if keep_line:
                        os.makedirs(save_folder_keepline, exist_ok=True)
                    
                    gt_root = line_root.replace("line", "gt")
                    for gt_path in glob(osp.join(gt_root, "*.png")):
                        # process the ground truth
                        json_path = gt_path.replace("gt", "seg").replace("png", "json")
                        shutil.copy(gt_path, save_folder)
                        shutil.copy(json_path, save_folder)
                        print(gt_path, "is given.")
                        if keep_line:
                            line_path=gt_path.replace("gt", "line")
                            merged_img_save_path= gt_path.replace("gt", character_name+'_keepline')
                            merge_color_line(line_path,gt_path,merged_img_save_path)
                        
                _, name_str_ref = osp.split(test_data["file_name_ref"][0])
                json_path_ref = osp.join(save_folder, name_str_ref + ".json")
                color_dict = load_json(json_path_ref)
                json_save_path = osp.join(save_folder, name_str + ".json")

                match_tensor = self.model(self.dis_data_to_cuda(test_data))
                match_scores = match_tensor["match_scores"].cpu().numpy()

                color_next_frame = {}
                unmatch_color = [0] * len(list(color_dict.values())[0])
                for i, scores in enumerate(match_scores):
                    color_lookup = np.array([(color_dict[str(i + 1)] if str(i + 1) in color_dict else unmatch_color) for i in range(len(scores))])
                    unique_colors = np.unique(color_lookup, axis=0)
                    accumulated_probs = [np.sum(scores[np.all(color_lookup == color, axis=1)]) for color in unique_colors]
                    color_next_frame[str(i + 1)] = unique_colors[np.argmax(accumulated_probs)].tolist()
                dump_json(color_next_frame, json_save_path)

                label_path = osp.join(character_root, "seg", name_str + ".png")
                img_save_path = json_save_path.replace(".json", ".png")
                colorize_label_image(label_path, json_save_path, img_save_path)

                if keep_line:
                    line_path=osp.join(save_path, 'line', name_str + ".png")
                    merged_img_save_path= osp.join(save_folder_keepline, name_str + ".png")
                    merge_color_line(line_path,img_save_path,merged_img_save_path)
                    print("image saved at", merged_img_save_path)