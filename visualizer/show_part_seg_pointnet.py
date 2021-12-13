import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import torch
import torch.nn.parallel
import torch.utils.data
from utils import to_categorical
from data_utils.ShapeNetDataLoader import PartNormalDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from show3d_balls import showpoints
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--idx', type=int, default=0, help='model index')
parser.add_argument('--dataset', type=str, default='', help='dataset path')
parser.add_argument('--class_choice', type=str, default=None, help='class choice',nargs='+')

opt = parser.parse_args()
print(opt)
class_choice = opt.class_choice
if class_choice is None:
    class_choice = []

root = '../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
d = PartNormalDataset(root=root, npoints=2048, split='test', normal_channel=True)
loader = torch.utils.data.DataLoader(d, batch_size=2, shuffle=False, num_workers=4)

idx = opt.idx
point,cls,seg = d[idx]



cmap = plt.cm.get_cmap("hsv",50)
cmap = np.array([cmap(i) for i in range(50)])[:,:3]
gt = cmap[seg -1,:]

model_name = 'pointnetplusplus_msg'
MODEL = importlib.import_module(model_name)
classifier = MODEL.get_model().cuda()
checkpoint = torch.load('../log/part_seg/pointnet2_part_seg_msg/checkpoints/best_model.pth')
classifier.load_state_dict(checkpoint['model_state_dict'])

# state_dict = torch.load(opt.model)
# classifier = PointNetDenseCls(k= state_dict['conv4.weight'].size()[0])
# classifier.load_state_dict(state_dict)
classifier.eval()   
    

for batch_id, (points, label, target) in tqdm(enumerate(loader), total=len(loader),
                                                      smoothing=0.9):
    batchsize, num_point, _ = points.size()
    cur_batch_size, NUM_POINT, _ = points.size()
    points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
    points = points.transpose(2, 1)
    points, label, target = points.cuda(), label.squeeze().cuda(), target.cuda()
    
    seg_pred, _ = classifier(points, to_categorical(label, 16))
    
    seg_pred = seg_pred.contiguous().view(-1, 16)
    pred_choice = seg_pred.data.max(1)[1]

    print(pred_choice)
    
    pred_color = cmap[pred_choice[0].cpu().detach().numpy(), :]
    
    print(pred_color.shape)
    showpoints(point[:,:3], gt, pred_color)
