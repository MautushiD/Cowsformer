import torch


def resize_bbox(bbox, target_size):
    """
    turn a bbox from one size to another

    param
    ---
    bbox: all floating numbers: [x, y, w, h]
    target_size: a tensor of [h, w] ***** very important! *****

    return
    ---
    a new bbox with the same aspect ratio, but resized to fit the target size
    """
    bx, by, bw, bh = bbox
    th, tw = target_size
    new_x = bx * tw
    new_y = by * th
    new_w = bw * tw
    new_h = bh * th
    return torch.tensor([new_x, new_y, new_w, new_h])


def xywh2xyxy(xywh, in_xy="center", img_size=None):
    """
    param
    ---
    xywh: all floating numbers: [x, y, w, h] (range: 0-1)
    in_xy: "center" (yolo) or "top-left" (coco) of the input bbox xy
        - Note that the output from DETR is in "center" format
    img_size: (w, h) of the image
    return
    ---
    xyxy: all floating numbers: [x1, y1, x2, y2] from top-left to bottom-right
    """
    x, y, w, h = xywh
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)
    if in_xy == "center":
        dw, dh = w / 2, h / 2
        x1 = x - dw
        y1 = y - dh
        x2 = x + dw
        y2 = y + dh
    elif in_xy == "top-left":
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
    if img_size:
        img_w, img_h = img_size
        x1, x2 = x1 * img_w, x2 * img_w
        y1, y2 = y1 * img_h, y2 * img_h
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    return torch.tensor([x1, y1, x2, y2])
