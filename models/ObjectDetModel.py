from torch import nn 

class Custom_predictor(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(Custom_predictor,self).__init__()
        self.additional_layer = nn.Linear(in_channels,in_channels) #this is the additional layer  
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 24)

    def forward(self,x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        x = self.additional_layer(x)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


class ObjDet(nn.Module):
    def __init__(self, encoder,numcls):
        super().__init__()
        self.encoder = encoder
        self.output_channels = 512
        self.objdetLayers = Custom_predictor(self.output_channels,numcls)
    def forward(self, inputs):
        output = self.encoder(inputs)
        scores, bboxes = self.objdetLayers(output)
        return scores, bboxes