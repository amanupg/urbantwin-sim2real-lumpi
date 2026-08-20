"""Run trained OpenPCDet model on the 50 LUMPI detection-test frames.

Runs ON THE TRAINING VM (needs pcdet + GPU). Produces predictions.json in the
challenge format. Frames are raw float32 Nx3 bins from the starter kit,
staged at gs://eccv-data-bucket-amlc/lumpi_test_frames.

Usage (on VM):
  python3 infer_lumpi.py --ckpt <path.pth> \
      --cfg OpenPCDet/tools/cfgs/custom_models/pointpillar_lumpi.yaml \
      --frames ~/lumpi_test_frames --out predictions.json \
      [--score_thresh 0.15]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# our own checkpoints are trusted; torch>=2.6 defaults to weights_only=True
# which rejects pcdet's pickled numpy scalars
_orig_torch_load = torch.load
torch.load = lambda *a, **k: _orig_torch_load(*a, **{**k, "weights_only": False})

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class BinFrameDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, frame_dir):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names,
                         training=False, root_path=Path(frame_dir), logger=None)
        self.files = sorted(Path(frame_dir).glob("*.bin"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        pts = np.fromfile(self.files[index], dtype=np.float32).reshape(-1, 3)
        input_dict = {"points": pts, "frame_id": self.files[index].stem}
        return self.prepare_data(data_dict=input_dict)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default="predictions.json")
    ap.add_argument("--score_thresh", type=float, default=0.15)
    args = ap.parse_args()

    cfg_from_yaml_file(args.cfg, cfg)
    logger = common_utils.create_logger()
    ds = BinFrameDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, args.frames)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda().eval()

    preds = {}
    with torch.no_grad():
        for i in range(len(ds)):
            data_dict = ds.collate_batch([ds[i]])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            p = pred_dicts[0]
            boxes = p["pred_boxes"].cpu().numpy()
            scores = p["pred_scores"].cpu().numpy()
            labels = p["pred_labels"].cpu().numpy()
            out = []
            for b, s, l in zip(boxes, scores, labels):
                if s < args.score_thresh:
                    continue
                out.append({
                    "box": [round(float(v), 3) for v in b[:7]],
                    "score": round(float(s), 4),
                    "class": cfg.CLASS_NAMES[int(l) - 1],
                })
            preds[ds.files[i].stem] = out
            print(f"{ds.files[i].stem}: {len(out)} dets")

    Path(args.out).write_text(json.dumps(preds))
    n = sum(len(v) for v in preds.values())
    print(f"wrote {args.out}: {len(preds)} frames, {n} detections")


if __name__ == "__main__":
    main()
