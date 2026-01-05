import torch
from src.training.evaluation import match_predictions

# one ground truth box
gt_boxes = torch.tensor([[0, 0, 100, 100]], dtype=torch.float32)

# three predictions
pred_boxes = torch.tensor(
    [
        [0, 0, 100, 100],     # perfect
        [10, 10, 90, 90],     # overlaps but duplicate
        [200, 200, 300, 300] # no overlap
    ],
    dtype=torch.float32,
)

pred_scores = torch.tensor([0.9, 0.8, 0.3])

tp, fp, fn = match_predictions(
    pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5
)

print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
