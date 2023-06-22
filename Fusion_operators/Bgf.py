#Billaterla guided fusion
import torch
import torch.nn as nn
class GBF:
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
                num_filters  = 3 * 2**i
                self.bev_fc1.append(nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, KernelS, stride=1, padding=pds, bias=False),
                    nn.BatchNorm2d(num_filters),
                    #nn.ReLU(inplace=True)
                ))

                self.cyl_fc1 .append(nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, KernelS, stride=1, padding=pds, bias=False),
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
                    nn.Conv2d(ch_in, num_filters, KernelS, stride=1, padding=pds, bias=False),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(inplace=True),
                ))
                self.fusion_transform_MFB .append(nn.Sequential(
                    nn.Conv2d(ch_in, num_filters, KernelS, stride=1, padding=pds, bias=False),
                    nn.BatchNorm2d(num_filters)
                ))
            

                self.combine_trans.append( nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, KernelS, stride=1, padding=pds, bias=False),
                    nn.BatchNorm2d(num_filters)
                    #nn.ReLU(inplace=True)
                ))




    def trad_fusion(self, RGB, LIDAR,nm):
            cyl_x1 = self.cyl_fc1[nm](RGB)
            bev_x1 = self.bev_fc1[nm](LIDAR)

            att_cyl_to_bev = torch.sigmoid(self.cyl_att_path[nm](cyl_x1))
            att_bev_to_cyl = torch.sigmoid(self.bev_att_path[nm](bev_x1))

            cyl_x2 = self.cyl_fc2[nm](cyl_x1)
            bev_x2 = self.bev_fc2[nm](bev_x1)

            pt_cyl_pre_fusion = (cyl_x1 + att_bev_to_cyl * cyl_x2)
            pt_bev_pre_fusion = (bev_x1 + att_cyl_to_bev * bev_x2)

            point_features = torch.cat([pt_cyl_pre_fusion, pt_bev_pre_fusion], dim=1)

            conv_features = self.fusion_transform[nm](point_features)
            #conv_features = conv_features.squeeze(0).transpose(0, 1)
            return conv_features