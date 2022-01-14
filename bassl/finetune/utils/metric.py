"""
- Recall / Recall@3s / AP
"""

import json
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torchmetrics
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

__DEBUG__ = False


class F1ScoreMetric(torchmetrics.classification.F1):
    def __init__(self, **metric_args):

        metrics_args = {"compute_on_step": False, "dist_sync_on_step": True}

        super().__init__(**metrics_args)


class AveragePrecisionMetric(torchmetrics.classification.AveragePrecision):
    """
    ref:
        - https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/classification/average_precision.py
    """

    def __init__(self, **metric_args):

        metrics_args = {"compute_on_step": False, "dist_sync_on_step": True}

        super().__init__(**metrics_args)


class SklearnAPMetric(torchmetrics.Metric):
    def __init__(self, **metric_args):

        metrics_args = {"compute_on_step": False, "dist_sync_on_step": True}

        super().__init__(**metrics_args)
        self.add_state("prob", default=[], dist_reduce_fx="cat")
        self.add_state("gts", default=[], dist_reduce_fx="cat")

    def update(self, prob, gts):
        assert isinstance(prob, torch.FloatTensor) or isinstance(
            prob, torch.cuda.FloatTensor
        )
        assert isinstance(gts, torch.LongTensor) or isinstance(
            gts, torch.cuda.LongTensor
        )

        self.prob.append(prob)
        self.gts.append(gts)

    def compute(self) -> torch.Tensor:
        prob = self.prob.cpu().numpy()
        gts = self.gts.cpu().numpy()
        ap = average_precision_score(np.nan_to_num(gts), np.nan_to_num(prob))
        precision, recall, thresholds = precision_recall_curve(
            np.nan_to_num(gts), np.nan_to_num(prob)
        )
        ap = torch.Tensor([ap]).type_as(self.prob)
        precision = torch.Tensor([precision]).type_as(self.prob)
        recall = torch.Tensor([recall]).type_as(self.prob)
        return ap, precision, recall


class SklearnAUCROCMetric(torchmetrics.Metric):
    def __init__(self, **metric_args):

        metrics_args = {"compute_on_step": False, "dist_sync_on_step": True}

        super().__init__(**metrics_args)
        self.add_state("prob", default=[], dist_reduce_fx="cat")
        self.add_state("gts", default=[], dist_reduce_fx="cat")

    def update(self, prob, gts):
        assert isinstance(prob, torch.FloatTensor) or isinstance(
            prob, torch.cuda.FloatTensor
        )
        assert isinstance(gts, torch.LongTensor) or isinstance(
            gts, torch.cuda.LongTensor
        )

        self.prob.append(prob)
        self.gts.append(gts)

    def compute(self) -> torch.Tensor:
        prob = self.prob.cpu().numpy()
        gts = self.gts.cpu().numpy()
        auc = roc_auc_score(np.nan_to_num(gts), np.nan_to_num(prob))
        fpr, tpr, threshold = roc_curve(np.nan_to_num(gts), np.nan_to_num(prob))

        auc = torch.Tensor([auc]).type_as(self.prob)
        fpr = torch.Tensor([fpr]).type_as(self.prob)
        tpr = torch.Tensor([tpr]).type_as(self.prob)
        return auc, fpr, tpr


class AccuracyMetric(torchmetrics.classification.Accuracy):
    """
    ref:
        - https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/classification/accuracy.py
        - https://github.com/PyTorchLightning/metrics/blob/f61317ca17e3facc16e09c0e6cef0680961fc4ff/torchmetrics/functional/classification/accuracy.py#L72
    """

    def __init__(self, **metric_args):

        metrics_args = {"compute_on_step": False, "dist_sync_on_step": True}

        super().__init__(**metrics_args)

        self.eps = 1e-5
        self.threshold = 0.5
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, prob, labels):

        assert isinstance(prob, torch.FloatTensor) or isinstance(
            prob, torch.cuda.FloatTensor
        )
        assert isinstance(labels, torch.LongTensor) or isinstance(
            labels, torch.cuda.LongTensor
        )

        gt_one = labels == 1
        gt_zero = labels == 0
        pred_one = prob >= self.threshold
        pred_zero = prob < self.threshold

        self.tp += (gt_one * pred_one).sum()
        self.fp += (gt_zero * pred_one).sum()
        self.tn += (gt_zero * pred_zero).sum()
        self.fn += (gt_one * pred_zero).sum()

    def compute(self) -> Dict[str, torch.Tensor]:
        # compute final result
        tp = self.tp
        fp = self.fp
        tn = self.tn
        fn = self.fn

        assert (tp + fn) > 0
        assert (fp + tn) > 0

        output = {}
        output["acc1"] = 100.0 * tp / (tp + fn + self.eps)
        output["acc0"] = 100.0 * tn / (fp + tn + self.eps)
        output["acc"] = 100.0 * (tp + tn) / (tp + fn + fp + tn + self.eps)

        if __DEBUG__:
            self.print(
                f"TP:{tp.item()} / FP:{fp.item()} / TN:{tn.item()} / FN:{fn.item()}"
            )

        return output


class MovieNetMetric(torchmetrics.Metric):
    def __init__(self):
        super().__init__(compute_on_step=False, dist_sync_on_step=True)
        self.add_state("vidx_data", default=[], dist_reduce_fx="cat")
        self.add_state("sid_data", default=[], dist_reduce_fx="cat")
        self.add_state("pred_data", default=[], dist_reduce_fx="cat")
        self.add_state("gt_data", default=[], dist_reduce_fx="cat")

        self.vid2idx = json.load(
            open(
                "/data/project/rw/workspace_jason/boundary_detection/data/movienet_data/anno/vid2idx.json",
                "r",
            )
        )
        self.idx2vid = {idx: vid for vid, idx in self.vid2idx.items()}
        self.shot_path = "/data/project/rw/MovieNet/anno_scene318/shot_movie318"

    def update(self, vid, sid, pred, gt):
        # assert isinstance(vid, torch.Tensor)

        self.vidx_data.append(pred.new_tensor(self.vid2idx[vid], dtype=torch.long))
        self.sid_data.append(pred.new_tensor(int(sid), dtype=torch.long))
        self.pred_data.append(pred)
        self.gt_data.append(gt)

    def compute(self) -> torch.Tensor:

        result = defaultdict(dict)
        for vidx, sid, pred, gt in zip(
            self.vidx_data, self.sid_data, self.pred_data, self.gt_data
        ):
            result[self.idx2vid[vidx.item()]][sid.item()] = {
                "pred": pred.item(),
                "gt": gt.item(),
            }

        # compute exact recall
        recall = self._compute_exact_recall(result)
        recall_at_second = self._compute_recall_at_second(result)
        miou = self._compute_mIoU(result)

        del result  # recall, recall_one, pred, gt, preds, gts
        torch.cuda.empty_cache()
        return recall, recall_at_second, miou

    def _compute_exact_recall(self, result):
        recall = []
        for _, result_dict_one in result.items():
            preds, gts = [], []
            for _, item in result_dict_one.items():
                pred = int(item.get("pred"))
                gt = int(item.get("gt"))
                preds.append(pred)
                gts.append(gt)

            recall_one = recall_score(gts, preds, average="binary")
            recall.append(recall_one)
        # print('Recall: ', np.mean(recall))

        recall = np.mean(recall)
        pt_recall = self.pred_data[0].new_tensor(recall, dtype=torch.float)

        del recall, recall_one, pred, gt, preds, gts
        return pt_recall

    def _compute_recall_at_second(self, result, num_neighbor_shot=5, threshold=3):

        recall = []
        for vid, result_dict_one in result.items():
            shot_fn = "{}/{}.txt".format(self.shot_path, vid)
            with open(shot_fn, "r") as f:
                shot_list = f.read().splitlines()

            cont_one, total_one = 0, 0
            for shotid, item in result_dict_one.items():
                gt = int(item.get("gt"))
                shot_time = int(shot_list[int(shotid)].split(" ")[1])
                if gt != 1:
                    continue

                total_one += 1
                for ind in range(0 - num_neighbor_shot, 1 + num_neighbor_shot):
                    # shotid_cp = self.strcal(shotid, ind)
                    shotid_cp = shotid + ind
                    if shotid_cp < 0 or (shotid_cp >= len(shot_list)):
                        continue
                    shot_time_cp = int(shot_list[shotid_cp].split(" ")[1])
                    item_cp = result_dict_one.get(shotid_cp)
                    if item_cp is None:
                        continue
                    else:
                        pred = item_cp.get("pred")
                        # FPS == 24
                        gap_time = np.abs(shot_time_cp - shot_time) / 24
                        if gt == pred and gap_time < threshold:
                            cont_one += 1
                            break

            recall_one = cont_one / (total_one + 1e-5)
            recall.append(recall_one)

        recall = np.mean(recall)
        pt_recall = self.pred_data[0].new_tensor(recall, dtype=torch.float)
        return pt_recall

    def _compute_mIoU(self, result):
        mious = []
        for vid, result_dict_one in result.items():
            shot_fn = "{}/{}.txt".format(self.shot_path, vid)
            with open(shot_fn, "r") as f:
                shot_list = f.read().splitlines()

            gt_dict_one, pred_dict_one = {}, {}
            for shotid, item in result_dict_one.items():
                gt_dict_one.update({shotid: item.get("gt")})
                pred_dict_one.update({shotid: item.get("pred")})
            gt_pair_list = self.get_pair_list(gt_dict_one)
            pred_pair_list = self.get_pair_list(pred_dict_one)
            if pred_pair_list is None:
                mious.append(0)
                continue
            gt_scene_list = self.get_scene_list(gt_pair_list, shot_list)
            pred_scene_list = self.get_scene_list(pred_pair_list, shot_list)
            if gt_scene_list is None or pred_scene_list is None:
                return None
            miou1 = self.cal_miou(gt_scene_list, pred_scene_list)
            miou2 = self.cal_miou(pred_scene_list, gt_scene_list)
            mious.append(np.mean([miou1, miou2]))

        mious = np.mean(mious)
        pt_miou = self.pred_data[0].new_tensor(mious, dtype=torch.float)
        return pt_miou

    def get_scene_list(self, pair_list, shot_list):
        scene_list = []
        if pair_list is None:
            return None
        for item in pair_list:
            start = int(shot_list[int(item[0])].split(" ")[0])
            end = int(shot_list[int(item[-1])].split(" ")[1])
            scene_list.append((start, end))
        return scene_list

    def cal_miou(self, gt_scene_list, pred_scene_list):
        mious = []
        for gt_scene_item in gt_scene_list:
            rats = []
            for pred_scene_item in pred_scene_list:
                rat = self.getRatio(pred_scene_item, gt_scene_item)
                rats.append(rat)
            mious.append(np.max(rats))
        miou = np.mean(mious)
        return miou

    def getRatio(self, interval_1, interval_2):
        interaction = self.getIntersection(interval_1, interval_2)
        if interaction == 0:
            return 0
        else:
            return interaction / self.getUnion(interval_1, interval_2)

    def getIntersection(self, interval_1, interval_2):
        assert interval_1[0] < interval_1[1], "start frame is bigger than end frame."
        assert interval_2[0] < interval_2[1], "start frame is bigger than end frame."
        start = max(interval_1[0], interval_2[0])
        end = min(interval_1[1], interval_2[1])
        if start < end:
            return end - start
        return 0

    def getUnion(self, interval_1, interval_2):
        assert interval_1[0] < interval_1[1], "start frame is bigger than end frame."
        assert interval_2[0] < interval_2[1], "start frame is bigger than end frame."
        start = min(interval_1[0], interval_2[0])
        end = max(interval_1[1], interval_2[1])
        return end - start

    def get_pair_list(self, anno_dict):
        sort_anno_dict_key = sorted(anno_dict.keys())
        tmp = 0
        tmp_list = []
        tmp_label_list = []
        anno_list = []
        anno_label_list = []
        for key in sort_anno_dict_key:
            value = anno_dict.get(key)
            tmp += value
            tmp_list.append(key)
            tmp_label_list.append(value)
            if tmp == 1:
                anno_list.append(tmp_list)
                anno_label_list.append(tmp_label_list)
                tmp = 0
                tmp_list = []
                tmp_label_list = []
                continue
        if len(anno_list) == 0:
            return None
        while [] in anno_list:
            anno_list.remove([])
        tmp_anno_list = [anno_list[0]]
        pair_list = []
        for ind in range(len(anno_list) - 1):
            cont_count = int(anno_list[ind + 1][0]) - int(anno_list[ind][-1])
            if cont_count > 1:
                pair_list.extend(tmp_anno_list)
                tmp_anno_list = [anno_list[ind + 1]]
                continue
            tmp_anno_list.append(anno_list[ind + 1])
        pair_list.extend(tmp_anno_list)
        return pair_list
