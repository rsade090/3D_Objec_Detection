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
transform = transforms.Compose([
  transforms.Resize((256,256)),])
name2idx = cnf.CLASS_NAME_TO_ID
idx2name = {v:k for k, v in name2idx.items()}

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

    def draw_rect(self, img, corners, labels, filename):
        img3 = img.numpy().copy()
        for i in range(corners.shape[0]):
            minx = min(corners[i][:,0]).numpy()
            miny = min(corners[i][:,1]).numpy()
            maxx = max(corners[i][:,0]).numpy()
            maxy = max(corners[i][:,1]).numpy()
            img3 = cv2.rectangle(img3, (int(minx), int(miny)), (int(maxx), int(maxy)), (255,0,0),1)
            cv2.putText(img3, idx2name[int(labels[i])], (int(minx), int(miny)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 1)
        cv2.imwrite(filename+'.jpg', img3)       
        return


    def random_crop_image_Bbox(self):
        BATCH_SIZE = 1
        NUM_BOXES = 5
        IMAGE_HEIGHT = 256
        IMAGE_WIDTH = 256
        CHANNELS = 3
        CROP_SIZE = (24, 24)
        image = tf.random.normal(shape=(
        BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS) )
        boxes = tf.random.uniform(shape=(NUM_BOXES, 4))
        box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0,
        maxval=BATCH_SIZE, dtype=tf.int32)
        output = tf.image.crop_and_resize(image, boxes, box_indices, CROP_SIZE)
        print(output.shape)
        return    

    def randPosCrop(self, imagee, bboxes, labels, bevs, fovs):
        images = []
        allbboxes = []
        alllabels = []
        allbevs = []
        allfovs = []
        count = 0
        for boxes in bboxes:
            image = imagee[count]
            label = labels[count]
            fov = fovs[count]
            bev = bevs[count]

            count += 1
            for j in range(boxes.shape[0]):
                posxy = torch.randint(256, (1, 2))
                xpos, ypos = 127, 127 #posxy[0][0], posxy[0][1] 
                minx = min(boxes[j][:,0])
                miny = min(boxes[j][:,1])
                maxx = max(boxes[j][:,0])
                maxy = max(boxes[j][:,1])
                boxcx = (maxx - minx)/2 + minx
                boxcy = (maxy - miny)/2 + miny
                boximg = image[int(miny): int(maxy), int(minx): int(maxx)]
                #do randCrop
                cropminx = max(boxcx - xpos, 0)
                cropminy = max(boxcy - ypos, 0)
                cropmaxx = cropminx + 256
                cropmaxy = cropminy + 256 
                mimg = image[int(cropminy): min(int(cropmaxy), image.shape[0]), int(cropminx): min(int(cropmaxx), image.shape[1])].clone()
                if mimg.shape[0] <= 220 or mimg.shape[1] <= 220:
                    continue
                # calc the position of each box after crop
                mm = transform(torch.permute(mimg, (2, 0, 1)))
                mm = torch.permute(mm, (1, 2, 0))
                #mm = cv2.resize(mimg.numpy(), dsize=(256,256), interpolation=cv2.INTER_LINEAR)
                images.append(mm)
                tmpboxes = []
                tmplabels = []
                for i in range(boxes.shape[0]):
                    curbox = boxes[i].clone()
                    curminx = min(curbox[:,0])
                    curminy = min(curbox[:,1])
                    curmaxx = max(curbox[:,0])
                    curmaxy = max(curbox[:,1])     
                    if curmaxx <= cropminx or curminy >= cropmaxy or curminx >= cropmaxx or curmaxy <= cropminy:
                        continue
                    curbox[:,0] = (curbox[:,0] - cropminx ) * (256/mimg.shape[1])
                    for f in range(8):
                        if curbox[f,0] < 0 : 
                            curbox[f,0] = 0
                        if curbox[f,0] > 256: 
                            curbox[f,0] = 256
                    curbox[:,1] = (curbox[:,1] - cropminy ) * (256/mimg.shape[0])
                    for f in range(8):
                        if curbox[f,1] < 0 : 
                            curbox[f,1] = 0
                        if curbox[f,1] > 256 : 
                            curbox[f,1] = 256
                    tmpboxes.append(curbox)
                    tmplabels.append(label[i])
                    
                tmpboxes = torch.stack(tmpboxes)
                tmplabels = torch.stack(tmplabels)
                self.draw_rect(mm, tmpboxes, tmplabels, 'afterCropBoxes')
                                
                allbboxes.append(tmpboxes) 
                alllabels.append(tmplabels) 
                allbevs.append(bev)     
                allfovs.append(fov)    

        #self.draw_rect(images[0], allbboxes[0], alllabels[0], 'afterAllCropBoxes')
        allbboxes = pad_sequence(allbboxes, batch_first=True, padding_value=-1)
        alllabels = pad_sequence(alllabels, batch_first=True, padding_value=-1)
        return torch.stack(images), torch.stack(allbevs) , torch.stack(allfovs), allbboxes, alllabels

def main():
    print("some samples for testing augmentation methods")
    dataAug = DataAugmentation()
    #dataAug.random_crop_image_Bbox()
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
        dataAug.draw_rect(img[0],targetBox[0],targetLabel[0], "beforeCrop")
        img_n, bev_n, fov_n, targetBox_n, targetLabel_n = dataAug.randPosCrop(img,targetBox,targetLabel, bev, fov)
 

if __name__ =="__main__":
    main()