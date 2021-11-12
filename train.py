#Need it to work with directories
import os 
#Need for train/set data division
import part_dataset_all_normal

#Initialization of variables
NUM_POINT=2048

#initialize root directory
Base_Dir = os.path.dirname(os.path.abspath(__file__))
Root_Dir = os.path.dirname(Base_Dir)


# Shapenet official train/test split
DATA_PATH = os.path.join(Root_Dir, 'data', 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
TRAIN_DATASET = part_dataset_all_normal.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='trainval')
TEST_DATASET = part_dataset_all_normal.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='test')