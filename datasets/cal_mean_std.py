from PIL import Image
import numpy as np

# from config import *

# IMG_DIR1 = '/zzp/fashionAI/fashionAI_attributes_train1/Images/pant_length_labels/'
# IMG_DIR2 = '/zzp/fashionAI/fashionAI_attributes_train2/Images/pant_length_labels/'
IMG_DIR = '/workspace/fashionai/datasets/fashionAI/train/Images/sleeve_length_labels/'


def check_files(file_list1, file_list2):
    for i in file_list1:
        if i in file_list2:
            print("oh no!")
        else:
            continue
    print("no file is same")


def get_files(dir):
    import os
    if not os.path.exists(dir):
        return []
    if os.path.isfile(dir):
        return [dir]
    result = []
    for subdir in os.listdir(dir):
        sub_path = os.path.join(dir, subdir)
        result += get_files(sub_path)
    return result


r = 0  # r mean
g = 0  # g mean
b = 0  # b mean

r_2 = 0  # r^2
g_2 = 0  # g^2
b_2 = 0  # b^2

total = 0

# file1 = get_files(IMG_DIR1)
# file2 = get_files(IMG_DIR2)
files = get_files(IMG_DIR)
count = len(files)

for i, image_file in enumerate(files):
    print('Process: {} / {}'.format(i, count), end="\r", flush=True)
    img = Image.open(image_file)
    # img = img.resize((299, 299))
    img = np.asarray(img)
    img = img.astype('float32') / 255.
    total += img.shape[0] * img.shape[1]

    r += img[:, :, 0].sum()
    g += img[:, :, 1].sum()
    b += img[:, :, 2].sum()

    r_2 += (img[:, :, 0] ** 2).sum()
    g_2 += (img[:, :, 1] ** 2).sum()
    b_2 += (img[:, :, 2] ** 2).sum()

r_mean = r / total
g_mean = g / total
b_mean = b / total

r_var = r_2 / total - r_mean ** 2
g_var = g_2 / total - g_mean ** 2
b_var = b_2 / total - b_mean ** 2

print('Mean is %s' % ([r_mean, g_mean, b_mean]))
print('Var is %s' % ([r_var, g_var, b_var]))

#################################################
task_dict = {
    'coat_length_labels': 8,  # 衣长 25774
    'collar_design_labels': 5,  # 领子 17452
    'lapel_design_labels': 5,  # 翻领 15910
    'neck_design_labels': 5,  # 脖颈 13850
    'neckline_design_labels': 10,  # 颈线 33524
    'pant_length_labels': 6,  # 裤子 21463
    'skirt_length_labels': 6,  # 裙子 21778
    'sleeve_length_labels': 9,  # 袖子 30584
}

mean_dict = {
    'coat_length_labels': [0.671, 0.636, 0.624],  # 上衣
    'collar_design_labels': [0.646, 0.607, 0.592],  # 衣领
    'lapel_design_labels': [0.633, 0.595, 0.585],  # 翻领
    'neck_design_labels': [0.633, 0.588, 0.568],  # 脖子
    'neckline_design_labels': [0.643, 0.601, 0.584],  # 领口
    'pant_length_labels': [0.652, 0.627, 0.615],  # 裤子
    'skirt_length_labels': [0.648, 0.614, 0.602],  # 裙子
    'sleeve_length_labels': [0.675, 0.637, 0.622],  # 袖子
}

std_dict = {
    'coat_length_labels': [0.099, 0.106, 0.108],  # 上衣
    'collar_design_labels': [0.084, 0.087, 0.088],  # 衣领
    'lapel_design_labels': [0.089, 0.094, 0.095],  # 翻领
    'neck_design_labels': [0.081, 0.084, 0.086],  # 脖子
    'neckline_design_labels': [0.084, 0.087, 0.089],  # 领口
    'pant_length_labels': [0.101, 0.101, 0.101],  # 裤子
    'skirt_length_labels': [0.095, 0.100, 0.101],  # 裙子
    'sleeve_length_labels': [0.098, 0.104, 0.105],  # 袖子
}
my_mean_dict = {
    'coat_length_labels': [0.652, 0.616, 0.603],  # 上衣
    'collar_design_labels': [0.642, 0.602, 0.588],  # 衣领
    'lapel_design_labels': [0.633, 0.594, 0.582],  # 翻领
    'neck_design_labels': [0.634, 0.587, 0.568],  # 脖子
    'neckline_design_labels': [0.645, 0.602, 0.587],  # 领口
    'pant_length_labels': [0.649, 0.623, 0.611],  # 裤子
    'skirt_length_labels': [0.644, 0.609, 0.602],  # 裙子
    'sleeve_length_labels': [0.664, 0.625, 0.613],  # 袖子 mean 0.645 0.607 0.594
}


my_std_dict = {
    'coat_length_labels': [0.093, 0.099, 0.101],  # 上衣
    'collar_design_labels': [0.084, 0.088, 0.088],  # 衣领
    'lapel_design_labels': [0.088, 0.092, 0.093],  # 翻领
    'neck_design_labels': [0.081, 0.085, 0.087],  # 脖子
    'neckline_design_labels': [0.084, 0.088, 0.090],  # 领口
    'pant_length_labels': [0.084, 0.088, 0.088],  # 裤子
    'skirt_length_labels': [0.090, 0.095, 0.096],  # 裙子
    'sleeve_length_labels': [0.093, 0.097, 0.099],  # 袖子 std 0.087 0.081 0.093
}

###############################################################
# mean=[0.485, 0.456, 0.406]
# std=[0.229, 0.224, 0.225]

# coat
# [0.6516631843165972, 0.6157289742500424, 0.6026083054662622]
# [0.09346615936095837, 0.09916823311540673, 0.10062731825945698]
# collar
# [0.6424705542494645, 0.6022679179552431, 0.5880025122063092]
# [0.08418099068819124, 0.08750965056267418, 0.08849591152789232]

# pant
# [0.6486975088350596, 0.6225290142351277, 0.6111255716307032]
# [0.0945333578573943, 0.09520872534093971, 0.09602502085325487]

# neckline
# [0.6456620573312923, 0.6022456080966028, 0.5866874603986827]
# [0.08425778422661712, 0.08816728175925315, 0.08977195441823438]

# skirt
# [0.6438393337381889, 0.6089977034194829, 0.5990414714035148]
# [0.09025221828459418, 0.09459761653455784, 0.09572068888626462]

# sleeve
# [0.6635670394522036, 0.6254083904605464, 0.6125041850254153]
# [0.09261010533587888, 0.09749894377836976, 0.09904537748302827]
