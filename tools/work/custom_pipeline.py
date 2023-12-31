from typing import Any
from mmseg.datasets import PIPELINES
import torch, os  
import numpy as np
from PIL import Image 
import random
import torchvision.transforms as transforms  
import matplotlib.pyplot as plt
import torchvision


class RandomBlackValues(object):
    """
    Class that fills -1.0 values in image with uniform random [0, 1).
    """
    def __call__(self, image):
        
        mask_size = (int(image.size(1) * 0.3), int(image.size(2) * 0.3))  # Adjust the mask size
        mask = torch.ones(1, image.size(1), image.size(2))  # Initialize with all ones
        x = random.randint(0, image.size(1) - mask_size[0])
        y = random.randint(0, image.size(2) - mask_size[1]) 
        mask[0, x:x + mask_size[0], y:y + mask_size[1]] = 0  # Set the mask region to 0

        # Apply the mask to the image
        image = image * mask
        
        return image

## perturbing cityscapes
@PIPELINES.register_module()
class CityTransform:
    def __init__(self) -> None:
        # self.img_norm_cfg = dict(
        # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        self.mean = (0.485, 0.456, 0.406) 
        self.std = (0.229, 0.224, 0.225)
        
        
    def __call__(self, results):
        
        # torch.manual_seed(0) # reproducibility of results  # not required, random erasing is getting fixed by it
        # input = torch.tensor(np.array(Image.open(os.path.join(results['filename']))), dtype=torch.float32)
        input = Image.open(os.path.join(results['filename']))
        # input = results['img'] ## since, some transforms is already used
        data_transforms = transforms.Compose([
            # transforms.TrivialAugmentWide(),  ## not compatible with torch 1.8 version! 
            transforms.ToTensor(), 
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.RandomErasing(p =1.0, value=0), ## getting grayish color, so making my own black values mask
            # transforms.RandomCrop(size=(1024,1024), pad_if_needed = True) ## only doing on image,but required to be executed both image and its label!
            # transforms.ColorJitter(brightness=(0.5,1.5),contrast=(0.5,1)), ## to many hyper-params to deal with  
            # transforms.RandAugment(num_ops= 4), # not compatible with torch 1.8 version! 
            # transforms.RandomErasing(scale=(0.02, 0.4), value=-1),  
            # RandomBlackValues(), 
        ])
        results['img'] = data_transforms(input)
        
        ########## saving the perturbed cityscapes images
        ########## in numpy fashion  ##### to large to save ~3k images
        # result_img_arr = results['img'].numpy()
        # id_name = results['filename'].split('/')[-1].replace('.png', '.npy')
        # root_folder = results['filename'].split('train')[0]
        # save_path = os.path.join(root_folder, 'custom_train', id_name)        
        # np.save(save_path, result_img_arr)
        ######### to save in '.png' format 
        # Save path
        id_name = results['filename'].split('/')[-1]
        root_folder = results['filename'].split('train')[0] ## change 'val' to 'train' later
        save_path = os.path.join(root_folder, 'custom_train', id_name) 
        
        # ### saving the image as it is, with just de-standardising and converting to range without normalising the image ## results are more darker here 
        # torchvision.utils.save_image(results['img'], save_path) 

        ##### saving the image by normalising it
        # Normalize the standardized tensor to the range [0, 1] 
        # normalized_tensor = (results['img'] - results['img'].min()) / (results['img'].max() - results['img'].min())
        # # Convert the PyTorch tensor to a NumPy array
        # numpy_array = (normalized_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # Assuming CHW format (channels, height, width)
        # # Create a PIL image from the NumPy array
        # image = Image.fromarray(numpy_array)
        # # Save the image as a PNG file
        # id_name = results['filename'].split('/')[-1]
        # root_folder = results['filename'].split('train')[0]
        # save_path = os.path.join(root_folder, 'custom_train', id_name) 
        # image.save(save_path)
        ## save as above code (" saving the image by normalising it")
        torchvision.utils.save_image(results['img'], save_path, normalize = True) 

        return results
        



# @PIPELINES.register_module()
# class MyTransform: 
#     def __init__(self, num_masks = 1000,patch_size = 20, mask_main_path = '/home/sidd_s/scratch/dataset/random_bin_masks/', img_size_h = 1024, img_size_w = 1024):
#         self.num_masks = num_masks 
#         self.patch_size = patch_size
#         self.mask_main_path = mask_main_path 
#         self.img_size_h = img_size_h 
#         self.img_size_w = img_size_w
    
#     def __call__(self, results): 
#         transforms_compose_label = transforms.Compose([
#                         transforms.Resize((self.img_size_h, self.img_size_w), interpolation=Image.NEAREST)])  # to have same aspect ration thus changing h,w dimensions
#         gt_color_path = os.path.join(results['seg_prefix'], results['ann_info']['seg_map']).replace('_gt_labelTrainIds','_gt_labelColor') 
#         mask_paths = []  
#         mask_lst = os.listdir(self.mask_main_path)      
#         for i in range(self.num_masks): 
#             rand = random.randint(0, 19) #inclusive  
#             mask_paths.append(os.path.join(self.mask_main_path, mask_lst[rand]))
#         per_gt = grp_perturb_gt_gen(mask_paths, gt_color_path, pred_path = results['filename'], patch_size=self.patch_size)      
#         results['img'] = np.array(transforms_compose_label(per_gt), dtype="float32") 
#         results['img'] = results['img'] / 255.0  
     
#         return results 

# @PIPELINES.register_module()
# class MyValTransform: 
    
#     def __init__(self, val_perturb_path = '/home/sidd_s/scratch/dataset/acdc_trainval/rgb_anon/night/synthetic/val/1000n_20p_dannet_pred'):
#         self.val_perturb_path = val_perturb_path
    
#     def __call__(self, results): 
#         ## perform the validation transform that is needed  
        
#         label_perturb_path = os.path.join(self.val_perturb_path, results['ori_filename']).replace('_rgb_anon.png','_gt_labelColor.png')  
#         # label_perturb_path = os.path.join(self.val_perturb_path, results['ori_filename'])  
#         results['img'] = np.array(Image.open(label_perturb_path), dtype="float32")  
#         results['img'] = results['img']  / 255.0
#         return results
    

# def grp_perturb_gt_gen(mask_paths, gt_path, pred_path, patch_size):
#     gt = Image.open(gt_path)  
#     pred = Image.open(pred_path) 
#     gt = np.array(gt)
#     pred = np.array(pred) 
#     big_mask = np.zeros((gt.shape[0], gt.shape[1]))
#     for mask_path in mask_paths:
#         mask =  Image.open(mask_path).convert('L')
#         mask = np.array(mask.resize((patch_size, patch_size), Image.NEAREST)) # strange clouds of dimensions
#         randx = np.random.randint(gt.shape[0]-patch_size)
#         randy = np.random.randint(gt.shape[1]-patch_size) 
#         big_mask[randx: randx+patch_size, randy: randy+patch_size] = mask  

#     gt[big_mask==255] = pred[big_mask==255] 
#     per_gt = Image.fromarray(gt)
#     return per_gt


"""
# Extra module might be useful later

########## TODO one class structure learnng ############ 

@PIPELINES.register_module()
class MyTransform_binary: 
    def __init__(self, num_masks = 1000,patch_size = 20, mask_main_path = '/home/sidd_s/scratch/dataset/random_bin_masks/', img_size_h = 1024, img_size_w = 1024):
        self.num_masks = num_masks 
        self.patch_size = patch_size
        self.mask_main_path = mask_main_path 
        self.img_size_h = img_size_h 
        self.img_size_w = img_size_w
    
    def __call__(self, results): 
        transforms_compose_label = transforms.Compose([
                        transforms.Resize((self.img_size_h, self.img_size_w), interpolation=Image.NEAREST)])  # to have same aspect ration thus changing h,w dimensions
        gt_color_path = os.path.join(results['seg_prefix'], results['ann_info']['seg_map']).replace('_gt_labelTrainIds','_gt_labelColor') 
        mask_paths = []  
        mask_lst = os.listdir(self.mask_main_path)      
        for i in range(self.num_masks): 
            rand = random.randint(0, 19) #inclusive  
            mask_paths.append(os.path.join(self.mask_main_path, mask_lst[rand]))
        per_gt = grp_perturb_gt_gen(mask_paths, gt_color_path, pred_path = results['filename'], patch_size=self.patch_size)      
        results['img'] = np.array(transforms_compose_label(per_gt), dtype="float32") 
        results['img'] = results['img'] / 255.0      
        return results 

@PIPELINES.register_module()
class MyValTransform_binary: 
    
    def __init__(self, val_perturb_path = '1000n_20p_dannet_pred'):
        self.val_perturb_path = val_perturb_path
    
    def __call__(self, results): 
        ## perform the validation transform that is needed  
        
        label_perturb_path = os.path.join('/home/sidd_s/scratch/dataset','acdc_trainval/rgb_anon/night/synthetic/val', self.val_perturb_path, results['ori_filename']).replace('_rgb_anon.png','_gt_labelColor.png')  
        results['img'] = np.array(Image.open(label_perturb_path), dtype="float32")  
        results['img'] = results['img'] / 255.0  
        
        return results
    
def grp_perturb_gt_gen_perturb(mask_paths, gt_path, pred_path, patch_size):
    gt = Image.open(gt_path)  
    pred = Image.open(pred_path) 
    gt = np.array(gt)
    pred = np.array(pred) 
    big_mask = np.zeros((gt.shape[0], gt.shape[1]))
    for mask_path in mask_paths:
        mask =  Image.open(mask_path).convert('L')
        mask = np.array(mask.resize((patch_size, patch_size), Image.NEAREST)) # strange clouds of dimensions
        randx = np.random.randint(gt.shape[0]-patch_size)
        randy = np.random.randint(gt.shape[1]-patch_size) 
        big_mask[randx: randx+patch_size, randy: randy+patch_size] = mask  

    gt[big_mask==255] = pred[big_mask==255]   
    
    ## forming binary mask 
    for i in range(20):
        if i == 14: 
            continue
        else:
            gt[gt==i] = 255  
    per_gt = Image.fromarray(gt)
    return per_gt
    
    """