from typing import Any
from mmseg.datasets import PIPELINES
import torch, os  
import numpy as np
from PIL import Image 
import random
import torchvision.transforms as transforms  


## perturbing cityscapes
@PIPELINES.register_module()
class CityTransform:
    def __init__(self) -> None:
        pass 
        
    def __call__(self, results):
        
        print('****************', results, results['ori_filename'])
        input = Image.open(os.path.join(results['filename'])) 
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        results['img'] = data_transforms(input)
    
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