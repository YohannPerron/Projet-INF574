import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_layers import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()

        self.SetAbstract1 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.1, 0.2, 0.4], nsample_list=[32, 64, 128], input_size=6, NN_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.SetAbstract2 = PointNetSetAbstractionMsg(npoint=128, radius_list=[0.4,0.8], nsample_list=[64, 128], input_size=128+128+64, NN_list=[[128, 128, 256], [128, 196, 256]])
        self.SetAbstract3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, input_size=512 + 3, NNlayers=[256, 512, 1024], group_all=True)

        self.FeatProp3 = PointNetFeaturePropagation(input_size=1536, NNLayers=[256, 256])
        self.FeatProp2 = PointNetFeaturePropagation(input_size=576, NNLayers=[256, 128])
        self.FeatProp1 = PointNetFeaturePropagation(input_size=150+3, NNLayers=[128, 128])

        self.Conv1 = nn.Conv1d(128, 128, 1)
        self.Batch1 = nn.BatchNorm1d(128)

        self.D_Out = nn.Dropout(0.5)
        self.Conv2 = nn.Conv1d(128, 50, 1)

    def forward(self, xyz, cls_label):
        B,C,N = xyz.shape

        #Abstraction
        new_xyz_1, new_points_concat_1 = self.SetAbstract1(xyz[:,:3,:], xyz)
        new_xyz_2, new_points_concat_2 = self.SetAbstract2(new_xyz_1, new_points_concat_1)
        new_xyz_3, new_points_concat_3 = self.SetAbstract3(new_xyz_2, new_points_concat_2)

        # Feature Propagation
        new_points_concat_2 = self.FeatProp3(new_xyz_2, new_xyz_3, new_points_concat_2, new_points_concat_3)
        new_points_concat_1 = self.FeatProp2(new_xyz_1, new_xyz_2, new_points_concat_1, new_points_concat_2)


        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        xyz = self.FeatProp1(xyz[:,:3,:], new_xyz_1, torch.cat([cls_label_one_hot,xyz[:,:3,:],xyz],1), new_points_concat_1)

        # Fully connected 
        res=self.Conv1(xyz)
        res=self.Batch1(res)
        res = F.relu(res)

            #Apply DropOut
        res = self.D_Out(res)

        res = self.Conv2(res)
        res = F.log_softmax(res, dim=1)
        res = res.permute(0, 2, 1)

        return res, new_points_concat_3


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss