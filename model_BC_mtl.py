from model_resnet_unet50 import UNetWithResnet50Encoder
import torch
import torch.nn as nn

class BC_mtl(nn.Module):
    def __init__(self,num_classes=1):
        super(BC_mtl, self).__init__()
        self.backbone = UNetWithResnet50Encoder()
        self.head = BC_head(num_classes=num_classes)

        self.criterion = nn.MSELoss(reduction='sum')

    def forward(self, x, train = True, ground_truth = []):
        x = self.backbone(x)
        object_pred, parts_pred = self.head(x)
        if train:
            object_true, parts_true = ground_truth[0], ground_truth[1]

            # Change model outputs tensor order
            object_pred = object_pred.permute(0, 2, 3, 1)

            object_loss = self.criterion(object_pred, object_true)
            parts_loss = self.criterion(parts_pred[0], parts_true)*0.33
            total_loss = object_loss + parts_loss

            return object_pred, parts_pred, object_loss, parts_loss, total_loss
        else:
            return object_pred, parts_pred


class BC_head(nn.Module):
    def __init__(self, num_classes, channel=64, bn_momentum=0.1):
        super(BC_head, self).__init__()
        # probmap_cluster
        self.object_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
            #nn.Sigmoid()
        )
        self.parts_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
            #nn.Sigmoid()
        )
    
    def forward(self, x):
        object_map = self.object_head(x)
        parts_map = self.parts_head(x)

        return object_map,parts_map

if __name__=='__main__':
    model = BC_mtl(num_classes=2).cuda()
    input = torch.rand((2, 3, 512, 512)).cuda()
    object_pred, parts_pred = model(input, train = False)
    print(object_pred.shape, parts_pred.shape)