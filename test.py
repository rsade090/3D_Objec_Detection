from torchvision import transforms
import cv2
from Kitti_Process.kittiDataset import KittiDataset
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from tqdm import tqdm
import torch
import tensorflow as tf
import config.kitti_config as cnf
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import config.kitti_config as kittiCnf
from models.fasterRCNN import *
transform = transforms.Compose([
  transforms.Resize((256,256)),])
name2idx = cnf.CLASS_NAME_TO_ID
idx2name = {v:k for k, v in name2idx.items()}
import numpy as np

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = \
            box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                # Filter target_boxes by pred_label so that we only match against boxes of our own label
                filtered_target_position, filtered_targets = zip(*filter(lambda x: target_labels[x[0]] == pred_label, enumerate(target_boxes)))

                # Find the best matching target for our predicted box
                iou, box_filtered_index = bbox_iou(pred_box.unsqueeze(0), torch.stack(filtered_targets)).max(0)

                # Remap the index in the list of filtered targets for that label to the index in the list with all targets.
                box_index = filtered_target_position[box_filtered_index]

                # Check if the iou is above the min treshold and i
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics

def main():
    print("some samples for testing augmentation methods")
    out_c, out_h, out_w = 256, 16, 16 #256, 12, 40 #2048, 15, 20
    width_scale_factor = configs.imageSize[1] // out_w
    height_scale_factor = configs.imageSize[0] // out_h
    out_size = (out_h, out_w)
    name2idx = kittiCnf.CLASS_NAME_TO_ID
    idx2name = {v:k for k, v in name2idx.items()}
    n_classes = len(name2idx) - 1 # exclude pad idx
    roi_size = (2, 2)
    detector = TwoStageDetector(configs.imageSize, out_size, out_c, n_classes, roi_size)
    detector.load_state_dict(torch.load("/home/hooshyarin/Desktop/model100.pt"))#/home/hooshyarin/Documents/3D_Objec_Detection/model_weights/model38.pt"))
    configs = edict()
    configs.hm_size = (152, 152)
    configs.imageSize = (375,1242)
    configs.max_objects = 50
    configs.num_classes = 3
    configs.dataset_dir = "/home/hooshyarin/Documents/KITTI/"
    val_set = KittiDataset(configs, mode='val', lidar_aug=None, hflip_prob=0.)
    dataloader_test = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=val_set.collate_fn, num_workers=2, pin_memory=True)
    for data in tqdm(dataloader_test):
        img, bev, fov, targetBox, targetLabel = data
        imgs = (torch.permute(img, (0,3, 1, 2))).to('cuda', dtype=torch.float32)
        bevs = (torch.permute(bev, (0,3, 1, 2))).to('cuda', dtype=torch.float32)
        targetB = [v.to('cuda', dtype=torch.float32) for v in targetBox]
        targetL = [t.to('cuda', dtype=torch.int64) for t in targetLabel]
        detector.eval()
        proposals_final, conf_scores_final, classes_final = detector.inference(imgs, bevs, conf_thresh=0.99, nms_thresh=0.01) 
        proposals_final = pad_sequence(proposals_final, batch_first=True, padding_value=-1)
        prop_proj_1 = project_bboxes(proposals_final, width_scale_factor, height_scale_factor, mode='a2p')
        classes_pred_1 = [idx2name[cls] for cls in classes_final[0].tolist()]


    return    
if __name__ =="__main__":
    main()