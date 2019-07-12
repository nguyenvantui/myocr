import torch.utils.data
import torch
import numpy as np
import glob
import cv2
from torchvision import transforms
import xml.etree.ElementTree as xmlparse

transforms = transforms.Compose([transforms.ToTensor()])

# loading image and label to the dataset

class qdataset(torch.utils.data.Dataset):

    def __init__(self, split, default_shape, data_link, max_gt):
        self.split = split
        self.data_link = data_link
        self.shape = default_shape
        self.max_ground_truth = max_gt
        self.images = glob.glob(data_link+"/images/*.png")

    def __getitem__(self, index):

        link_image = self.images[index]
        link_label = self.images[index].replace("images","labels").replace("png","xml")
        img = cv2.imread(link_image)
        img, scale = self.process_img(img, self.shape)

        # return img, info, gt, num
        # info = torch.zeros([3])
        info = torch.tensor([img.shape[0], img.shape[1], scale])
        img = transforms(img)
        # 5 = xmin,ymin,xmax, ymax,label

        gt, num = self.get_label(link_label)

        # image, information of image, ground_truh_box, number_of_box
        return img, info, gt, num

    def process_img(self, im, target_size):

        # im = im.astype(np.float32, copy=False)
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        img = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation = cv2.INTER_AREA)
        img = img.astype(np.float32)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return img, im_scale

    def get_label(self, path):
        anno = xmlparse.parse(path)
        bbox = list()
        # name
        label = list()
        count = 0
        gt = torch.zeros(self.max_ground_truth, 5)
        for obj in anno.findall('object'):
            # print(obj)
            count += 1
            bndbox_anno = obj.find('bndbox')
            bbox.append([ int(bndbox_anno.find(tag).text) for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
            label.append(int(obj.find('name').text.lower()))
            # pass
        # bbox = np.stack(bbox).astype(np.float32)
        # label = np.stack(label).astype(np.int32)
        for idx in range(count):
            gt[idx,:] = torch.tensor([bbox[idx][0],bbox[idx][1],bbox[idx][2],bbox[idx][3],label[idx]])
        total = torch.tensor([count])
        return gt, total

    def __len__(self):
        return len(self.images)
