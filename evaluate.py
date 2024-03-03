import supervision as sv


def from_yolo(metrics):
    # source
    # ultralytics.utils.metrics.DetMetrics
    """
    args
    ----
        metrics: dict
            metrics from model.val()

    return
    ------
        json: dict
    """
    # metrics
    map5095 = metrics.box.map.round(4)
    map50 = metrics.box.map50.round(4)
    precision = metrics.box.p[0].round(4)
    recall = metrics.box.r[0].round(4)
    f1 = metrics.box.f1[0].round(4)
    # confusion matrix
    conf_mat = metrics.confusion_matrix.matrix  # conf=0.25, iou_thres=0.45
    n_all = conf_mat[:, 0].sum()
    n_fn = conf_mat[1, 0].sum()
    n_fp = conf_mat[0, 1].sum()
    # write json
    json_out = dict(
        map5095=map5095,
        map50=map50,
        precision=precision,
        recall=recall,
        f1=f1,
        n_all=int(n_all),
        n_fn=int(n_fn),  # false negative
        n_fp=int(n_fp),
    )
    return json_out


def from_sv(preds, obs):
    """
    both preds and obs are lists of sv.Detections
    """
    metric_matrix = compute_confusion_matrix(preds, obs)
    SV_MAP = sv.MeanAveragePrecision.from_detections(
        predictions=preds,
        targets=obs,
    )
    # write json
    json_out = dict(
        map5095=SV_MAP.map50_95,
        map50=SV_MAP.map50,
        precision=metric_matrix["precision"],
        recall=metric_matrix["recall"],
        f1=2
        * (metric_matrix["precision"] * metric_matrix["recall"])
        / (metric_matrix["precision"] + metric_matrix["recall"]),
        n_all=metric_matrix["tp"] + metric_matrix["fn"],
        n_fn=metric_matrix["fn"],
        n_fp=metric_matrix["fp"],
    )
    return json_out


def compute_confusion_matrix(
    preds,
    obs,
    conf_threshold=0.25,
    iou_threshold=0.45,
):
    confusion_matrix = sv.ConfusionMatrix.from_detections(
        predictions=preds,
        targets=obs,
        classes=["balloon"],
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    tp = confusion_matrix.matrix[0][0]
    fn = confusion_matrix.matrix[0][1]
    fp = confusion_matrix.matrix[1][0]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return dict(
        tp=int(tp),
        fn=int(fn),
        fp=int(fp),
        precision=precision,
        recall=recall,
    )
