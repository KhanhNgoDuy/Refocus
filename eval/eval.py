# %%
import pandas as pd
import cv2
from math import log10, sqrt 
import numpy as np 
from skimage.metrics import structural_similarity
# import lpips
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# %%
# def PSNR(img1, img2): 
#     mse = np.mean((img1 - img2) ** 2) 
#     if(mse == 0):  # MSE is zero means no noise is present in the signal . 
#                   # Therefore PSNR have no importance. 
#         return 100
#     max_pixel = 255.0
#     psnr = 20 * log10(max_pixel / sqrt(mse)) 
#     return psnr 

# %%
def SSIM(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = structural_similarity(img1_gray, img2_gray, full=True)
    return score

# %%
def LPIPS(img1, img2):
    img1_norm = 2 * (cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0) - 1
    img2_norm = 2 * (cv2.cvtColor(img2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0) - 1
    
    img1_norm = torch.from_numpy(np.expand_dims(img1_norm, axis=0)).permute(0, 3, 1, 2)
    img2_norm = torch.from_numpy(np.expand_dims(img2_norm, axis=0)).permute(0, 3, 1, 2)
    
    # print(img1_norm.shape)
    # loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    # return loss_fn_alex(img1_norm, img2_norm)
    
    score_class_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex')
    score_class_vgg = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    score_class_squeeze = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
    return (score_class_alex(img1_norm, img2_norm).item(),
            score_class_vgg(img1_norm, img2_norm).item(),
            score_class_squeeze(img1_norm, img2_norm).item())


# %%

file_path = 'data.csv'
data = pd.read_csv(file_path)

ground_truth_folder_path = 'output_naive/'
output_folder_path = 'output_baseline/'

ground_truth_list = []
output_list = []
psnr_list = []
ssim_list = []
lpips_alex_list = []
lpips_vgg_list = []
lpips_squeeze_list = []

for index, row in data.iterrows():
    path = row['path'].replace('synthetic/', '')
    if 'f0' in path:
        continue
    
    ground_truth_image = cv2.imread(ground_truth_folder_path + path)
    output_image = cv2.imread(output_folder_path + path)

    psnr_value = cv2.PSNR(ground_truth_image, output_image)
    ssim_value = SSIM(ground_truth_image, output_image)
    lpips_value_tuple = LPIPS(ground_truth_image, output_image)

    # print(f'PSNR value = {psnr_value} dB')
    # print(f'SSIM value = {ssim_value}')
    # print(f'LPIPS value with backbone as (alex, vgg, squeeze)= {lpips_value}')
    # print('\n')

    ground_truth_list.append(ground_truth_folder_path + path)
    output_list.append(output_folder_path + path)
    psnr_list.append(psnr_value)
    ssim_list.append(ssim_value)
    lpips_alex_list.append(lpips_value_tuple[0])
    lpips_vgg_list.append(lpips_value_tuple[1])
    lpips_squeeze_list.append(lpips_value_tuple[2])
    
data = {
    'Ground truth file': ground_truth_list,
    'Output file': output_list,
    'PSNR (higher is better)': psnr_list,
    'SSIM (higher is better)': ssim_list,
    'LPIPS (backbone as AlexNet) (lower is better)': lpips_alex_list,
    'LPIPS (backbone as VGG) (lower is better)': lpips_vgg_list,
    'LPIPS (backbone as Squeeze) (lower is better)': lpips_squeeze_list,
}
df = pd.DataFrame(data)
csv_path = 'evaluation_result.csv'
df.to_csv(csv_path, index=False)
    


