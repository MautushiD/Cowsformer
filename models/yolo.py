import os
from ultralytics import YOLO

# local imports
from .device import get_device

# constants
ROOT = os.path.dirname(os.path.dirname(__file__))


class Niche_YOLO:
    def __init__(self, path_model="yolov8m.pt"):
        # attributes
        self.model = None
        self.dir_train = os.path.join(ROOT, "out", "train")
        self.dir_val = os.path.join(ROOT, "out", "val")

        # init
        self.load(path_model)

    def load(self, path_model):
        self.model = YOLO(path_model)
        print("model %s loaded" % path_model)

    def train(self, path_data, batch, dir_out):
        """
        args
        ----
            path_data: str
                path to data.yaml
            batch: int
                batch size
            name: str
                task folder name. Default "train_1"
        """
        self.model.train(
            data=path_data,
            batch=batch,
            device=get_device(),
            project=self.dir_train,
            name=dir_out,
            exist_ok=True,
        )
        best_model = os.path.join(self.dir_train, dir_out, "weights", "best.pt")
        self.load(best_model)

    def evaluate(self, dir_out, split="test"):
        metrics = self.model.val(
            split=split,
            device=get_device(),
            project=self.dir_val,
            name=dir_out,
            exist_ok=True,
        )
        # save metrics
        path_out = os.path.join(self.dir_val, dir_out, "results.json")
        with open(path_out, "w") as f:
            json = ext_metrics(metrics)
            json.dump(json, f)
        # return mAP50
        return json["map50"]


def ext_metrics(metrics):
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
    n_all = conf_mat[:, 0].sum().astype(int)
    n_missed = conf_mat[1, 0].sum().astype(int)
    n_false = conf_mat[0, 1].sum().astype(int)
    # write json
    json = dict(
        map5095=map5095,
        map50=map50,
        precision=precision,
        recall=recall,
        f1=f1,
        n_all=n_all,
        n_missed=n_missed,
        n_false=n_false,
    )
    return json
