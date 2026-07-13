import os

from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms




class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        
        self.root_dir =root_dir
        self.transform = transform 
        
        self.file_paths = list(os.listdir(root_dir))
        self.file_paths = [path for path in self.file_paths if path.endswith(('.jpg', '.jpeg', '.png'))] 

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.root_dir,self.file_paths[idx])
        # convert gray scale images to RGB
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image 
    

def get_transform(size, crop, final_size):

    transform_list = []

    if size>0:
        transform_list.append(transforms.Resize(size))

    if crop:
        transform_list.append(transforms.RandomCrop(final_size))

    else:
        transform_list.append(transforms.CenterCrop(final_size))

    
    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)


def adaptive_instance_normalization(content_feat, style_feat):
    # [batch size, channels, h, w]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    # [batch size, channels, h, w]
    size = feat.size()
    assert (len(size) == 4)
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std