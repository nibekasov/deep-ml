def iou(box_a, box_b):
    """
    box = [x1, y1, x2, y2]
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection box
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))

    inter_area = inter_w * inter_h

    # Individual areas
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    # Union
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def mean_average_precision(preds, gts, iou_thresh=0.5):
    """
    preds: [(class_id, score, box), ...]
    gts:   [(class_id, box), ...]
    """

    # Evaluate only classes that actually occur in ground truth
    classes = sorted(set(class_id for class_id, _ in gts))

    aps = []

    for cls in classes:

        # Ground truths for this class
        cls_gts = [
            box
            for class_id, box in gts
            if class_id == cls
        ]

        # Predictions for this class
        cls_preds = [
            (score, box)
            for class_id, score, box in preds
            if class_id == cls
        ]

        # Highest-confidence prediction first
        cls_preds.sort(key=lambda x: x[0], reverse=True)

        num_gt = len(cls_gts)

        # No predictions -> AP = 0
        if not cls_preds:
            aps.append(0.0)
            continue

        # Each GT may be matched only once
        used = [False] * num_gt

        tp = []
        fp = []

        for score, pred_box in cls_preds:

            best_iou = 0.0
            best_gt_idx = -1

            # Find best unmatched GT
            for i, gt_box in enumerate(cls_gts):

                if used[i]:
                    continue

                current_iou = iou(pred_box, gt_box)

                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = i

            # TP or FP?
            if best_gt_idx != -1 and best_iou >= iou_thresh:
                used[best_gt_idx] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        # Cumulative TP / FP
        tp_cum = []
        fp_cum = []

        tp_sum = 0
        fp_sum = 0

        for t, f in zip(tp, fp):
            tp_sum += t
            fp_sum += f

            tp_cum.append(tp_sum)
            fp_cum.append(fp_sum)

        # Precision / Recall
        precision = []
        recall = []

        for t, f in zip(tp_cum, fp_cum):
            precision.append(t / (t + f))
            recall.append(t / num_gt)

        # Precision envelope:
        # p[i] = max(p[i:])
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])

        # Integrate precision-recall curve
        ap = 0.0
        prev_recall = 0.0

        for p, r in zip(precision, recall):
            ap += (r - prev_recall) * p
            prev_recall = r

        aps.append(ap)

    if not aps:
        return 0.0

    return round(sum(aps) / len(aps), 4)
