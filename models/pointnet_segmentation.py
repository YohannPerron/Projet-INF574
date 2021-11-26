import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import STN3d, STNkd, feature_transform_reguliarzer


class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()

        #we are using normals therefore we have 6 normal channels
        nb_channel = 6
        #our number of classes relate to the number of parts in the object
        part_class = 50

        #set self attributes
        self.part_class = part_class

        #set up STN block 
        #It removes spatial invariance  by applying a learnable affine transformation followed by interpolation.
        self.tnet1 = STN3d(nb_channel)

        #set up conv and batch layers
        self.ConvClass1 = torch.nn.Conv1d(nb_channel, 64, 1)
        self.ConvClass2 = torch.nn.Conv1d(64, 128, 1)
        self.ConvClass3 = torch.nn.Conv1d(128, 128, 1)
        self.ConvClass4 = torch.nn.Conv1d(128, 512, 1)
        self.ConvClass5 = torch.nn.Conv1d(512, 2048, 1)


        self.Batchclass1 = nn.BatchNorm1d(64)
        self.Batchclass2 = nn.BatchNorm1d(128)
        self.Batchclass3 = nn.BatchNorm1d(128)
        self.Batchclass4 = nn.BatchNorm1d(512)
        self.Batchclass5 = nn.BatchNorm1d(2048)

        #the second pointnet is 128 instead of 64 again because we are using the normals
        self.tnet2 = STNkd(k=128)

        # segmentation part 
        #64+128+128+512+2048+2048+16 = 4944 relating to expand, out1,out2,out3,out4,out5
        self.ConvSeg1 = torch.nn.Conv1d(4944, 256, 1)
        self.ConvSeg2 = torch.nn.Conv1d(256, 256, 1)
        self.ConvSeg3 = torch.nn.Conv1d(256, 128, 1)
        self.ConvSeg4 = torch.nn.Conv1d(128, part_class, 1)

        self.BatchSeg1 = nn.BatchNorm1d(256)
        self.BatchSeg2 = nn.BatchNorm1d(256)
        self.BatchSeg3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud, label):

        #extract point cloud dimensions
        B, D, N = point_cloud.size()

        #apply the first tnet
        OutTnet = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)


        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = torch.bmm(point_cloud, OutTnet)
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)

        point_cloud = point_cloud.transpose(2, 1)


        #Apply first 3 convolutional layers with batching and activation function after each one
        out1 = F.relu(self.Batchclass1(self.ConvClass1(point_cloud)))
        out2 = F.relu(self.Batchclass2(self.ConvClass2(out1)))
        out3 = F.relu(self.Batchclass3(self.ConvClass3(out2)))

        #Apply second tnet
        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)


        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)


        #Apply the rest of convolutional layers with batching and activation function after each one again
        out4 = F.relu(self.Batchclass4(self.ConvClass4(net_transformed)))
        out5 = self.Batchclass5(self.ConvClass5(out4))

        #Apply max pooling
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 2048)

        out_max = torch.cat([out_max,label.squeeze(1)],1)
        expand = out_max.view(-1, 2048+16, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        net = F.relu(self.BatchSeg1(self.ConvSeg1(concat)))
        net = F.relu(self.BatchSeg2(self.ConvSeg2(net)))
        net = F.relu(self.BatchSeg3(self.ConvSeg3(net)))
        net = self.ConvSeg4(net)
        net = net.transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.part_class), dim=-1)
        net = net.view(B, N, self.part_class) # [B, N, 50]

        return net, trans_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss