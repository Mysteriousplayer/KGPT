from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class MyDataSet(Dataset):

    def __init__(self, select_data):
        super(MyDataSet, self).__init__()
        imgs = []
        raw_data = select_data['rgb']
        true_label = select_data['truelabel']
        pred_label = select_data['predlabel']
        for i in range(len(raw_data)):
            imgs.append(list((raw_data[i], true_label[i], pred_label[i])))

        imgs = np.array(imgs, dtype=object)
        self.imgs = imgs

    def __getitem__(self, item):
        img, targets, preds = self.imgs[item]
        try:
            img = Image.fromarray(np.uint8(img),'RGB')
        except:
            img_ = np.asarray(Image.open(img))
            try:
                img = Image.fromarray(np.uint8(img_), 'RGB')
            except:
                img_ = np.asarray(Image.open(img).convert('RGB'))
                img = Image.fromarray(np.uint8(img_), 'RGB')
        targets = int(targets)
        preds = int(preds)
        trans = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        img = trans(img)
        return img, targets, preds

    def __len__(self):
        return len(self.imgs)
