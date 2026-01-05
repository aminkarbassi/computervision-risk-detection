import torch


def compute_iou(box1, box2):
    """
    box: [xmin, ymin, xmax, ymax]
    returns IoU scalar
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0

    return inter_area / union_area


def match_predictions(
    pred_boxes,
    pred_scores,
    gt_boxes,
    iou_threshold=0.5,
):
    """
    Greedy matching for a single image.

    Returns:
        tp, fp, fn
    """
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    matched_gt = set()
    tp = 0
    fp = 0

    # sort predictions by confidence
    order = torch.argsort(pred_scores, descending=True)

    for idx in order:
        best_iou = 0.0
        best_gt = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue

            iou = compute_iou(
                pred_boxes[idx].tolist(),
                gt_box.tolist(),
            )

            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn
