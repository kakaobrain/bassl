"""
- kNN Precision
"""

from collections import defaultdict

import torch
import torchmetrics


class KnnPrecisionMetric(torchmetrics.Metric):
    def __init__(self, top_k_list):
        super().__init__(compute_on_step=False, dist_sync_on_step=True)
        self.add_state("feat_data", default=[], dist_reduce_fx=None)
        self.add_state("vids_data", default=[], dist_reduce_fx=None)
        self.add_state("scene_data", default=[], dist_reduce_fx=None)
        self.top_k_list = set(top_k_list)
        self.max_k = max(self.top_k_list)

    def update(self, vid, invideo_scene_id, feat):
        assert isinstance(invideo_scene_id, torch.Tensor)
        assert isinstance(vid, torch.Tensor)
        assert isinstance(feat, torch.Tensor)
        self.feat_data.append(feat)
        self.vids_data.append(vid)
        self.scene_data.append(invideo_scene_id)

    def compute(self) -> torch.Tensor:
        score = defaultdict(dict)
        pool_feats = defaultdict(list)
        pool_invideo_scene_id = defaultdict(list)
        pool_gts = defaultdict(dict)

        num_data = 0
        for vid, invideo_scene_id, gathered_feat in zip(
            self.vids_data, self.scene_data, self.feat_data
        ):
            vid = vid.item()
            invideo_scene_id = invideo_scene_id.item()
            if invideo_scene_id not in pool_gts[vid]:
                pool_gts[vid][invideo_scene_id] = set()
            pool_gts[vid][invideo_scene_id].add(len(pool_feats[vid]))
            pool_invideo_scene_id[vid].append(invideo_scene_id)
            pool_feats[vid].append(gathered_feat)
            num_data += 1

        for top_k in self.top_k_list:
            score[top_k] = {"correct": 0, "total": 0}

        for vid, gt in pool_gts.items():
            X = torch.stack(pool_feats[vid])
            sim = torch.matmul(X, X.t())
            sim = sim - 999 * torch.eye(sim.shape[0]).type_as(sim)  # exclude self
            indices = torch.argsort(sim, descending=True)
            assert indices.shape[1] >= self.max_k, f"{indices.shape[1]} >= {self.max_k}"
            indices = indices[:, : self.max_k]

            for j in range(indices.shape[0]):
                _cache = {"correct": 0, "total": 0}
                _query_scene_id = pool_invideo_scene_id[vid][j]
                for k in range(self.max_k):
                    if _query_scene_id in gt:
                        if indices[j][k].item() in gt[_query_scene_id]:
                            _cache["correct"] += 1
                    _cache["total"] += 1
                    if k + 1 in self.top_k_list and len(gt[_query_scene_id]) > k:
                        score[k + 1]["correct"] += _cache["correct"]
                        score[k + 1]["total"] += _cache["total"]

        for top_k in self.top_k_list:
            assert score[top_k]["total"] > 0
            score[top_k]["precision"] = (
                100.0 * score[top_k]["correct"] / score[top_k]["total"]
            )
        del X, sim, indices, pool_feats, pool_invideo_scene_id, pool_gts
        torch.cuda.empty_cache()
        return score
