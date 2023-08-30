
import os
import glob
import torch
import time
import math

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

data_transforms = {
    'aug': transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.95, 1.0)),  # Randomly crop and resize the image tensor
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally with a probability of 0.5
        # transforms.RandomVerticalFlip(p=0.5),  # Randomly flip the image vertically with a probability of 0.5
        transforms.RandomRotation(degrees=(-10, 10)),
    ]),
    'norm': transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),   
}
                
class ClickMe(Dataset):
    def __init__(self, file_paths, is_training=True):
        super(Dataset).__init__()
        self.file_paths = file_paths
        self.is_training = is_training
        
        
    def __getitem__(self, index):
        data = torch.load(self.file_paths[index])
        img, hmp, label = data['image'], data['heatmap'], data['label']
        
        img = img.to(torch.float32) / 255.0 # unit8 -> float32
        hmp = hmp.to(torch.float32) / 255.0 # unit8 -> float32
        label = label.to(torch.int64)       # int32 -> int64
        label = torch.squeeze(label)        # [batch_size, 1] -> [batch_size]
        
        if self.is_training:
            stacked_img = torch.cat((img, hmp), dim=0) 
            stacked_img = data_transforms['aug'](stacked_img) # Apply data augmentation
            img, hmp = stacked_img[:-1, :, :], stacked_img[-1:, :, :]
            # print(img.shape, hmp.shape) # torch.Size([3, 224, 224]) torch.Size([1, 224, 224])
            
        img = data_transforms['norm'](img)  # Apply ImageNet mean and std
        return img, hmp, label
                
    def __len__(self):
        return len(self.file_paths)
                    
if __name__ == "__main__":

    data_dir = "/media/data_cifs/pfeng2/Pseudo_ClickMe/Dataset/test/"
    train_file_paths = glob.glob(os.path.join(data_dir, 'PseudoClickMe/train/*.pth')) 
    print(train_file_paths[0])
    
    dataset = ClickMe(train_file_paths)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
    start = time.time()
    cnt = 0
    for imgs, hmps, labels in dataloader:
        # imgs, hmps, labels = preprocess(imgs, hmps, labels)
        
        if cnt == 0:
            print(imgs.shape, hmps.shape, labels.shape, labels.view(-1,1).shape)
            print(imgs[0].max(), imgs[0].min())
            print(imgs[0].dtype, hmps[0].dtype, labels[0].dtype)
        cnt += 1
        pass
        
    end = time.time()
    print(end-start) 
    

    
        