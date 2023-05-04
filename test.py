from torchvision import transforms
import cv2
from Kitti_Process.kittiDataset import KittiDataset
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from tqdm import tqdm
import torch
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
from dataAugmentation import DataAugmentation

def intersection_over_union(gt_box, pred_box):
    inter_box_top_left = [max(gt_box[0], pred_box[0]), max(gt_box[1], pred_box[1])]
    inter_box_bottom_right = [min(gt_box[2], pred_box[2]), min(gt_box[3], pred_box[3])]

    inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
    inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]

    intersection = inter_box_w * inter_box_h
    union = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) + (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])  - intersection
    
    iou = intersection / union

    return iou, intersection, union

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
    unique_classes1 = []
    for cls_row in target_cls:
        unique_classes1.extend(cls_row)
    unique_classes = torch.unique(torch.tensor(unique_classes1))
    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm(unique_classes, desc="Computing AP"):
        if c == -1:
            continue
        i = torch.tensor(pred_cls) == c
        n_gt = (torch.tensor(unique_classes1) == c).sum()  # Number of ground truth objects
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
            recall_curve = tpc / (n_gt)# + 1e-16)
            r.append(recall_curve[-1])
            if recall_curve[-1] > 1:
                print()
            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)
    print("P:", p)
    print("R:", r)
    print("AP:", ap)
    print("F1:", f1)  
    print("unique classes:",unique_classes)
    return p, r, ap, f1, unique_classes

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

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:][0] - box1[:][2] / 2, box1[:][0] + box1[:][2] / 2
        b1_y1, b1_y2 = box1[:][1] - box1[:][3] / 2, box1[:][1] + box1[:][3] / 2
        b2_x1, b2_x2 = box2[:][0] - box2[:][2] / 2, box2[:][0] + box2[:][2] / 2
        b2_y1, b2_y2 = box2[:][1] - box2[:][3] / 2, box2[:][1] + box2[:][3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[:][0], box1[:][1], box1[:][2], box1[:][3]
        b2_x1 = []    
        b2_y1 = []    
        b2_x2 = []    
        b2_y2 = []    
        for i in range(len(box2)):
            b2_x1.append(box2[i][0]) 
            b2_y1.append(box2[i][1]) 
            b2_x2.append(box2[i][2])
            b2_y2.append(box2[i][3])

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = []
    inter_rect_y1 = []
    inter_rect_x2 = []
    inter_rect_y2 = []
    for i in range(len(box2)):
        inter_rect_x1.append(torch.max(b1_x1, b2_x1[i]))
        inter_rect_y1.append(torch.max(b1_y1, b2_y1[i]))
        inter_rect_x2.append(torch.min(b1_x2, b2_x2[i]))
        inter_rect_y2.append(torch.min(b1_y2, b2_y2[i]))
    # Intersection area
    iouMax = 0
    iouIndex= 0
    for i in range(len(box2)):
        inter_area = (max(inter_rect_x2[i] - inter_rect_x1[i], 0)*max(inter_rect_y2[i] - inter_rect_y1[i], 0))
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2[i] - b2_x1[i] + 1) * (b2_y2[i] - b2_y1[i] + 1)
        iou  = (inter_area / (b1_area + b2_area - inter_area + 1e-16))
        if iouMax < iou:
            iouMax = iou
            iouIndex = i
    # inter_area = torch.clamp(np.array(inter_rect_x2.cpu()) - inter_rect_x1 + 1, min=0) * torch.clamp(
    #     inter_rect_y2 - inter_rect_y1 + 1, min=0
    # )
    # Union Area
    # b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    # b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    # iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iouMax, iouIndex

def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range((outputs[0].shape[0])):

        if outputs[0][sample_i] is None:
            continue
        output = outputs[0][sample_i]
        pred_boxes = output
        pred_scores = outputs[2][sample_i][:]
        pred_labels = outputs[1][sample_i][:]
        if (pred_labels.shape[0] == 0):
            continue
        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[0][sample_i] #[targets[:, 0] == sample_i][:, 1:]
        target_boxes= list()
        for i in range(annotations.shape[0]):
            minx = min(annotations[i,:,0])
            miny = min(annotations[i,:,1])
            maxx = max(annotations[i,:,0])
            maxy = max(annotations[i,:,1])
            target_boxes.append([minx,miny,maxx,maxy])
        #target_boxes = torch.stack(target_boxes,dim=0)
        target_labels =  targets[1][sample_i] #annotations[:, 0] if len(annotations) else []
        if (annotations.shape[0]):
            detected_boxes = []
            #target_boxes = annotations[:, 1:]
            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                if len(detected_boxes) == len(annotations):
                    break
                if pred_label not in target_labels:
                    continue
                # Filter target_boxes by pred_label so that we only match against boxes of our own label
                filtered_target_position, filtered_targets = zip(*filter(lambda x: target_labels[x[0]] == pred_label, enumerate(target_boxes)))
                # Find the best matching target for our predicted box
                iou, box_filtered_index = bbox_iou(pred_box, filtered_targets)
                # Remap the index in the list of filtered targets for that label to the index in the list with all targets.
                box_index = filtered_target_position[box_filtered_index]

                # Check if the iou is above the min treshold and i
                if iou*10 >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores.cpu(), pred_labels.cpu()])
    return batch_metrics

def draw_rect(img, corners, filename, labels):
    img3 = img[0].clone().detach().cpu().numpy()
    for i in range(corners.shape[1]):
        minx = min(corners[0][i][:,0]).numpy()
        miny = min(corners[0][i][:,1]).numpy()
        maxx = max(corners[0][i][:,0]).numpy()
        maxy = max(corners[0][i][:,1]).numpy()
        cv2.rectangle(img3, (int(minx), int(miny)), (int(maxx), int(maxy)), (255,0,0),1)
        cv2.putText(img3, labels[i], (int(minx), int(miny)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 1)
    cv2.imwrite(filename+'.jpg', img3)       
    return

def draw_rect2d(img, prop_proj_1, classes_pred_1):
    classes_pred_1 = [idx2name[cls] for cls in classes_pred_1[0].tolist()]
    img4 = (img[0].clone().detach()).cpu().numpy()
    for i in range(len(classes_pred_1)):
        img4 = cv2.rectangle(img4, (int(prop_proj_1[0][i][0]), int(prop_proj_1[0][i][1])), (int(prop_proj_1[0][i][2]), int(prop_proj_1[0][i][3])), (0,255,0),1)
        cv2.putText(img4, classes_pred_1[i], (int(prop_proj_1[0][i][0]), int(prop_proj_1[0][i][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 1)
    cv2.imwrite('afterTrainRect.jpg',img4) 

def main():
    print("some samples for testing augmentation methods")
    configs = edict()
    configs.hm_size = (152, 152)
    configs.imageSize = (375,1242)
    configs.max_objects = 50
    configs.num_classes = 3
    configs.dataset_dir = "/home/sadeghianr/Desktop/Datasets/Kitti/"
    out_c, out_h, out_w = 256, 16, 16 #256, 12, 40 #2048, 15, 20
    width_scale_factor = configs.imageSize[1] // out_w
    height_scale_factor = configs.imageSize[0] // out_h
    out_size = (out_h, out_w)
    name2idx = kittiCnf.CLASS_NAME_TO_ID
    idx2name = {v:k for k, v in name2idx.items()}
    n_classes = len(name2idx)# exclude pad idx
    roi_size = (2, 2)
    detector = TwoStageDetector(configs.imageSize, out_size, out_c, n_classes, roi_size).to('cuda')
    detector.load_state_dict(torch.load("/home/sadeghianr/Desktop/Codes/3D_Objec_Detection/model_weights/crop256_justRandomCroppedimage/model10.pt"))#/home/hooshyarin/Documents/3D_Objec_Detection/model_weights/model38.pt"))
    dataAug = DataAugmentation()
    val_set = KittiDataset(configs, mode='val', lidar_aug=None, hflip_prob=0.)
    dataloader_test = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=val_set.collate_fn, num_workers=2, pin_memory=True)
    sample_metrics = []
    targetnew = []
    for data in tqdm(dataloader_test):
        img, bev, fov, targetBox, targetLabel = data
        img, bev, fov, targetBox, targetLabel = dataAug.randPosCrop(img,targetBox,targetLabel, bev, fov)
        imgs = (torch.permute(img, (0,3, 1, 2))).to('cuda', dtype=torch.float32)
        bevs = (torch.permute(bev, (0,3, 1, 2))).to('cuda', dtype=torch.float32)
        targetB = [v.to('cuda', dtype=torch.float32) for v in targetBox]
        targetL = [t.to('cuda', dtype=torch.int64) for t in targetLabel]
        detector.eval()
        proposals_final, conf_scores_final, classes_final = detector.inference(imgs, bevs, conf_thresh=0.99, nms_thresh=0.1) 
        proposals_final = pad_sequence(proposals_final, batch_first=True, padding_value=-1)
        framelabels = [idx2name[cls] for cls in targetLabel[0].tolist()]
        draw_rect(img, targetBox, "MainImages", framelabels) # draw box before training
        im = img[0]
        pred_B = project_bboxes(proposals_final, width_scale_factor, height_scale_factor, mode='a2p')
        pred_L = classes_final #[idx2name[cls] for cls in classes_final[0].tolist()]
        pred_S = conf_scores_final
        draw_rect2d(img, pred_B, pred_L) #draw box after training
        print()
        # calc mAP
        sample_metrics += get_batch_statistics([pred_B, pred_L, pred_S], [targetB, targetL], 6.0)
        # Concatenate sample statistics
        targetnew += targetL
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, targetnew)
        print()
    return    
if __name__ =="__main__":
    main()