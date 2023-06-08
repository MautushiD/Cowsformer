"""
This py file is used to split the coco dataset into train, val, test set.
"""
import os
import json
import random


ROOT = os.path.dirname(__file__)
DIR_COCO = os.path.join(ROOT, "cow100", "coco")
PATH_JSON = os.path.join(DIR_COCO, "_annotations.coco.json")
LS_IMGS = [f for f in os.listdir(DIR_COCO) if f.endswith(".jpg")]
PROP_TRAIN = 0.64
PROP_VAL = 0.16


def main():
    coco_annot = read_coco(PATH_JSON)
    ls_id = get_coco_ids(coco_annot)
    id_train, id_val, id_test = split_ids(
        ls_id, prop_train=PROP_TRAIN, prop_val=PROP_VAL
    )

    dict_id = dict(
        train=id_train,
        val=id_val,
        test=id_test,
    )
    for split in dict_id:
        json_out = make_json(dict_id[split], coco_annot)
        save_json(json_out, split)


def read_coco(path_json) -> dict:
    with open(path_json, "r") as f:
        coco_annot = json.load(f)
    return coco_annot


def get_coco_ids(coco_annot):
    ls_id = [img["id"] for img in coco_annot["images"]]
    random.shuffle(ls_id)
    return ls_id


def split_ids(ls_id, prop_train=0.64, prop_val=0.16):
    # get counts
    n = len(ls_id)
    n_train = int(prop_train * n)
    n_val = int(prop_val * n)
    # split ids
    id_train = ls_id[:n_train]
    id_val = ls_id[n_train : n_train + n_val]
    id_test = ls_id[n_train + n_val :]
    return id_train, id_val, id_test


def make_json(image_ids: list, coco_annot: dict) -> dict:
    json_out = dict(
        info=coco_annot["info"],
        licenses=coco_annot["licenses"],
        images=[],
        annotations=[],
        categories=coco_annot["categories"],
    )
    for img in coco_annot["images"]:
        if img["id"] in image_ids:
            json_out["images"].append(img)
    for ann in coco_annot["annotations"]:
        if ann["image_id"] in image_ids:
            json_out["annotations"].append(ann)
    return json_out


def save_json(json_out, split):
    path_json = os.path.join(DIR_COCO, "_%s_annotations.coco.json" % split)
    with open(path_json, "w") as f:
        json.dump(json_out, f)


if __name__ == "__main__":
    main()
