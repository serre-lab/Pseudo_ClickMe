
import os
import glob
import time
import torch

if __name__ == "__main__":
    # compare(data_size=10, batch_size=32)
    
    root_dir = "mnt/disks/pseudo-clickme/"
    folders = ['Pseudo_ClickMe/train', 'Pseudo_ClickMe/val', 'ClickMe/train', 'ClickMe/val', 'ClickMe/test']
    for folder in folders:
        print("Processing folder: " + folder)
        
        save_dir = os.path.join(root_dir, 'dataset', folder)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        data_dir = os.path.join(root_dir, folder)
        paths = glob.glob(os.path.join(data_dir, '*.pth')) 
        
        start = time.time()
        cnt = 0
        file_cnt = 1
        n = len(paths)
        for path in paths:
            print("Processing %s | %s \r" % (str(file_cnt), str(n)), end="")
            data = torch.load(path)
            imgs, hmps, labels = data['images'], data['heatmaps'], data['labels']
            # print(imgs.shape, hmps.shape, labels.shape)
            labels = labels[:, None]
            for img, hmp, label in zip(imgs, hmps, labels):
                img = img[None, :, :, :]
                hmp = hmp[None, :, :, :]
                label = label[None, :]

                data = {
                    'image': img, # torch.Size([1, 3, 224, 224]) torch.uint8
                    'heatmap': hmp, # torch.Size([1, 1, 224, 224]) torch.uint8
                    'label': label, # torch.Size([1, 1]) torch.int32
                }
                
                # Save the data to a .pth file
                pth_path = os.path.join(save_dir, str(cnt) + ".pth")
                torch.save(data, pth_path)
                cnt += 1
                # break
                
            if os.path.exists(path):
                os.remove(path)
                
            file_cnt += 1
            
            # break
        print(" ")
        
        end = time.time()
        print(end-start) # 154.6498782634735
    

    
        