import os
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class FashionAIDataset(Dataset):
    AttrKey = {
        'coat_length_labels': 8,
        'pant_length_labels': 6,
        'skirt_length_labels': 6,
        'sleeve_length_labels': 9,
    }

    def __init__(self, data_root, attr_task, mode, transform=None):
        if attr_task not in ['coat_length_labels', 'collar_design_labels', 'lapel_design_labels', 'neck_design_labels',
                             'neckline_design_labels', 'pant_length_labels', 'skirt_length_labels',
                             'sleeve_length_labels']:
            print("{} attribute not exist!".format(attr_task))
            raise RuntimeError
        self.mode = mode
        self.transform = transform
        self.data_folder = data_root
        self.attr_task = attr_task
        if self.mode == "train":
            self.label_file = os.path.join(data_root, "Annotations", f"label_{self.attr_task}_train.csv")
        if self.mode == "test":
            self.label_file = os.path.join(data_root, "Annotations", f"label_{self.attr_task}_test.csv")

        self.df = pd.read_csv(self.label_file, header=1, names=['img_path', 'task', 'label'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 保证同一task
        img_info = self.df.iloc[idx]
        img_path = os.path.join(self.data_folder, img_info['img_path'])
        img_label = img_info['label']
        one_hot_target = np.zeros(FashionAIDataset.AttrKey[self.attr_task])
        # one_hot_target = F.one_hot(img_label, FashionAIDataset.AttrKey[self.attr_task])
        # todo: how to deal with label   nnnym
        y_label_idx = img_label.find('y')
        one_hot_target[y_label_idx] = 1
        # m_count = img_label.count("m")
        # if m_count != 0:
        #     m_idx_list = []
        #     m_idx=-1
        #     for i in range(m_count):
        #         m_idx = img_label.find("m", m_idx+1)
        #         m_idx_list.append(m_idx)
        #     if m_count == 1:
        #         one_hot_target[m_idx_list[0]] = 0.1
        #         one_hot_target[y_label_idx] = 0.9
        #     elif m_count == 2:
        #         one_hot_target[m_idx_list[0]] = 0
        #         one_hot_target[m_idx_list[1]] = 0.1
        #         one_hot_target[y_label_idx] = 0.9
        #     elif m_count == 3:
        #         one_hot_target[m_idx_list[0]] = 0
        #         one_hot_target[m_idx_list[1]] = 0.1
        #         one_hot_target[m_idx_list[2]] = 0.1
        #         one_hot_target[y_label_idx] = 0.9

        # use ablu toolkit
        # img_data = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

        # use randargument
        img_data = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            # albu toolkit
            # augmented = self.transform(image=img_data)
            # img_data = augmented['image']
            # torch
            img_data = self.transform(img_data)

        return img_data, one_hot_target

    # def get_filelist(self, file_dir):
    #     import os
    #     if not os.path.exists(file_dir):
    #         return []
    #     if os.path.isfile(file_dir):
    #         return [file_dir]
    #     result = []
    #     for subdir in os.listdir(file_dir):
    #         sub_path = os.path.join(file_dir, subdir)
    #         result += self.get_filelist(sub_path)
    #     return result


# class FashionAIPercatDataset(Dataset):
#     def __int__(self, data_root, task, transform):
#         self.data_root = data_root
#         self.img_folder1 = os.path.join(self.data_root, "Images")
#         self.img_list =
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#
#
#     def get_filelist(self, file_dir):
#         import os
#         if not os.path.exists(file_dir):
#             return []
#         if os.path.isfile(file_dir):
#             return [file_dir]
#         result = []
#         for subdir in os.listdir(file_dir):
#             sub_path = os.path.join(file_dir, subdir)
#             result += self.get_filelist(sub_path)
#         return result


if __name__ == '__main__':
    import albumentations as A
    from albumentations.pytorch import ToTensorV2


    def get_transforms(*, data):
        if data == 'train':
            return A.Compose([
                # A.Resize(CFG.size, CFG.size),
                A.RandomResizedCrop(512, 512),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.CoarseDropout(p=0.5),
                A.Cutout(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])

        elif data == 'test':
            return A.Compose([
                A.CenterCrop(512, 512),
                A.Resize(512, 512),
                # A.CenterCrop(CFG.size, CFG.size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])


    AttrKey = {
        'coat_length_labels': 8,  # 上衣
        'collar_design_labels': 5,  # 领子
        'lapel_design_labels': 5,  # 翻领
        'neck_design_labels': 5,  # 脖颈
        'neckline_design_labels': 10,  # 颈线
        'pant_length_labels': 6,  # 裤子
        'skirt_length_labels': 6,  # 裙子
        'sleeve_length_labels': 9,  # 袖子
    }
    ###########################################################################################################
    data_fold = '/workspace/fashionai/datasets/fashionAI/train'
    trainset = FashionAIDataset(data_fold, 'collar_design_labels', mode='train', transform=get_transforms(data='train'))
    train_loader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)

    for batch_idx, (data, label) in enumerate(train_loader):
        print(data.shape)
        print(label)
    #############################################################################################################

###########################################################################################################
# base_label = '/workspace/fashionai/datasets/fashionAI/train/Annotations/label_test.csv'
# for task in ["lapel_design_labels", "neck_design_labels",
#              "neckline_design_labels", "pant_length_labels", "skirt_length_labels",
#              "sleeve_length_labels"]:
#     df = pd.read_csv(base_label, header=None, names=['img_path', 'task', 'label'])
#     # train_label = f'/workspace/fashionai/datasets/fashionAI/train/Annotations/label_{task}_train.csv'
#     test_label = f'/workspace/fashionai/datasets/fashionAI/train/Annotations/label_{task}_test.csv'
#
#     df = df[df['task'].str.contains(task)]
#     df.to_csv(test_label, index=False)
#     print("{} done".format(task))
###########################################################################################################
# base_label = '/workspace/fashionai/datasets/fashionAI/train/Annotations/label.csv'
# train_label = '/workspace/fashionai/datasets/fashionAI/train/Annotations/label_train.csv'
# test_label = '/workspace/fashionai/datasets/fashionAI/train/Annotations/label_test.csv'
#
# df = pd.read_csv(base_label)
# shuffle_df = df.sample(frac=1, random_state=42)
# cut_idx = int(round(0.8 * shuffle_df.shape[0]))
# train_data, test_data = shuffle_df.iloc[:cut_idx], shuffle_df.iloc[cut_idx:]
# train_data.to_csv(train_label, index=False)
# test_data.to_csv(test_label, index=False)
#############################################################################################################

#############################################################################################################
# base_label2 = '/workspace/fashionai/datasets/fashionAI/train/Annotations/label2.csv'
# base_label1 = '/workspace/fashionai/datasets/fashionAI/train/Annotations/label1.csv'
# outputcsv = '/workspace/fashionai/datasets/fashionAI/train/Annotations/label11111.csv'
#
# df1 = pd.read_csv(base_label1)
# df2 = pd.read_csv(base_label2)
#
# all_csv = pd.concat([df1, df2], axis=0)
# all_csv.to_csv(outputcsv, index=False)
#
# check = pd.read_csv(outputcsv)
# print(check)
#############################################################################################################

#############################################################################################################
# with open(base_label, 'r') as f:
#     lines = f.readlines()
#     count = 1
#     for l in lines:
#         img_name = l.rstrip().split(',')[0].split('/')[-1]
#         tokens[img_name] = count
#         count += 1
#
# with open(label2, 'r') as F:
#     lines = F.readlines()
#     for l in lines:
#         img_name2 = l.rstrip().split(',')[0].split('/')[-1]
#         if img_name2 in tokens.keys():
#             print("find!!!!!! {}".format(img_name2))
#     print("data is clean...")
#############################################################################################################
