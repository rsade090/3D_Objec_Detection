        
import torch
import torch.nn as nn
class Fusion_ops:
    def __init__(self):
        
        pds=1
        KernelS =3
        self.bev_fc1 =nn.ModuleList()
        self.cyl_fc1 =nn.ModuleList()
        self.bev_fc2=nn.ModuleList()
        self.cyl_fc2=nn.ModuleList()
        self.bev_att_path =nn.ModuleList()
        self.cyl_att_path=nn.ModuleList()
        self.fusion_transform =nn.ModuleList()
        self.fusion_transform_MFB=nn.ModuleList()
        self.combine_trans =nn.ModuleList()
        #for i, num_blocks in enumerate(self.stage_blocks): 
        
        for i in range(1): 
                num_filters  =  6 * 2**i
                inoutfilter = 3 * 2**i
                self.bev_fc1.append(nn.Sequential(
                    nn.Conv2d(inoutfilter, num_filters, KernelS, stride=1, padding=pds, bias=False),
                    nn.BatchNorm2d(num_filters),
                    #nn.ReLU(inplace=True)
                ))

                self.cyl_fc1 .append(nn.Sequential(
                    nn.Conv2d(inoutfilter, num_filters, KernelS, stride=1, padding=pds, bias=False),
                    nn.BatchNorm2d(num_filters),
                    #nn.ReLU(inplace=True)
                ))

                self.bev_fc2 .append( nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, KernelS, stride=1, padding=pds, bias=False),
                    nn.BatchNorm2d(num_filters),
                ))

                self.cyl_fc2 .append( nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, KernelS, stride=1, padding=pds, bias=False),
                    nn.BatchNorm2d(num_filters),
                ))
            

                self.bev_att_path .append( nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, KernelS, stride=1, padding=pds),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_filters, num_filters, KernelS, stride=1, padding=pds),
                ))

                self.cyl_att_path .append(nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, KernelS, stride=1, padding=pds),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_filters, num_filters, KernelS, stride=1, padding=pds),
                ))

                ch_in = num_filters * 2
                self.fusion_transform .append( nn.Sequential(
                    nn.Conv2d(ch_in, inoutfilter, KernelS, stride=1, padding=pds, bias=False),
                    nn.BatchNorm2d(inoutfilter),
                    nn.ReLU(inplace=True),
                ))
                self.fusion_transform_MFB .append(nn.Sequential(
                    nn.Conv2d(ch_in, num_filters, KernelS, stride=1, padding=pds, bias=False),
                    nn.BatchNorm2d(num_filters)
                ))
            

                self.combine_trans.append( nn.Sequential(
                    nn.Conv2d(num_filters, inoutfilter, KernelS, stride=1, padding=pds, bias=False),
                    nn.BatchNorm2d(inoutfilter)
                    #nn.ReLU(inplace=True)
            ))
    def MFBv3(self, RGB, LIDAR,nm):
            cyl_x1 = self.cyl_fc1[nm](RGB)
            bev_x1 = self.bev_fc1[nm](LIDAR)

            att_cyl_to_bev = torch.sigmoid(self.cyl_att_path[nm](cyl_x1))
            att_bev_to_cyl = torch.sigmoid(self.bev_att_path[nm](bev_x1))

            cyl_x2 = self.cyl_fc2[nm](cyl_x1)
            bev_x2 = self.bev_fc2[nm](bev_x1)
    

            feat_aux = torch.add(att_cyl_to_bev, att_bev_to_cyl)
            inter = torch.mul(att_cyl_to_bev, att_cyl_to_bev)
            combined = self.combine_trans[nm](inter)

            point_features = torch.cat([combined, feat_aux], dim=1)
            out = self.fusion_transform_MFB[nm](point_features)

            out = torch.sqrt(nn.ReLU(inplace=True)(out)) + torch.sqrt(nn.ReLU(inplace=True)(-out))
            l2_normed = torch.nn.functional.normalize( out)  

            return l2_normed 