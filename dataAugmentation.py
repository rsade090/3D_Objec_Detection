from torchvision import transforms
import cv2
from Kitti_Process.kittiDataset import KittiDataset
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from tqdm import tqdm
import torch

class DataAugmentation():
    def __init__(self):
        self.crop_indices = 0
    def randCrop(self,img, bboxes):
        # find the upper left point and lower right point of the bb
        # ulx, uly = 0
        # lrx, lry = 0
        # for box in bboxes:
        #     minx = min(box[:,0])
        #     miny = min(box[0,:])
        #     maxx = max(box[:,0])
        #     maxy = max(box[0,:])
        #     if ulx > minx: ulx = minx
        #     if uly > miny: ulx = miny
        #     if ulx > minx: ulx = minx
        #     if ulx > minx: ulx = minx
        #img = torch.permute(img, (2, 0, 1))   
        self.crop_indices = transforms.RandomCrop.get_params(torch.permute(img, (2, 0, 1)) , output_size=(256, 256))
        i, j, h, w = self.crop_indices  
        mimg = img[i:i+h,j: j+h] 
        mbboxes = bboxes       
        return mimg, mbboxes
        
    def draw_rect(self, img, corners, filename):
        img3 = img[0]
        for i in range(corners.shape[1]):
            minx = min(corners[0][i][:,0]).numpy()
            miny = min(corners[0][i][:,1]).numpy()
            maxx = max(corners[0][i][:,0]).numpy()
            maxy = max(corners[0][i][:,1]).numpy()
            img3 = cv2.rectangle(img[0].cpu().numpy(), (int(minx), int(miny)), (int(maxx), int(maxy)), (255,0,0),1)
            #cv2.putText(img3, classes_pred_1[i], (int(minx), int(miny)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 1)
        cv2.imwrite(filename+'.jpg', img3)       
        return

def main():
    print("some samples for testing augmentation methods")
    dataAug = DataAugmentation()
    configs = edict()
    configs.hm_size = (152, 152)
    configs.imageSize = (375,1242)
    configs.max_objects = 50
    configs.num_classes = 3
    configs.dataset_dir = "/home/hooshyarin/Documents/KITTI/"
    val_set = KittiDataset(configs, mode='val', lidar_aug=None, hflip_prob=0.)
    dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=val_set.collate_fn, num_workers=2, pin_memory=True)
    for data in tqdm(dataloader_val):
        img, bev, fov, targetBox, targetLabel = data
        # show the image before the crop
        dataAug.draw_rect(img,targetBox, "beforeCrop")    
        mimg, mboxes = dataAug.randCrop(img[0], targetBox)
        # show the cropped image
        dataAug.draw_rect(mimg, mboxes, "afterCrop")    

if __name__ =="__main__":
    main()