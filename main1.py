import torch
import argparse
from datasets import qdataset
from tqdm import tqdm
from model.vgg_backbone import vgg_backbone
from model.resnet import resnet as resnet_fpn
from model.nms import nms
from model.detnet import detnet as detnet_fpn
from tools.dl import clip_gradient as clipping
from ipdb import set_trace as bp
import numpy as np
import cv2
import time

# hello world
# ==========================================================================================
parser = argparse.ArgumentParser(description='faster-rcnn')
parser.add_argument('--batch', dest='batch_size',
                    help='batch_size',
                    default=3, type=int)

parser.add_argument('--data_train', dest='data_train',
                    help='data_train',
                    default="../data/train/", type=str)

parser.add_argument('--data_test', dest='data_test',
                    help='data_test',
                    default="../data/test/", type=str)

parser.add_argument('--shape', dest='shape',
                    help='shape',
                    default=600, type=int)
parser.add_argument('--epochs', dest='epochs',
                    help='epochs',
                    default=10, type=int)

parser.add_argument('--display', dest='display',
                    help='display',
                    default=10, type=int)

parser.add_argument('--opt', dest='opt',
                    help='optimizer',
                    default="sgd", type=str)

parser.add_argument('--lr_max', dest='lr_max',
                    help='lr_max',
                    default=0.0001, type=float)

parser.add_argument('--lr_min', dest='lr_min',
                    help='lr_min',
                    default=0.00001, type=float)

parser.add_argument('--class', dest='z_class',
                    help='z_class',
                    default=10, type=int)

args = parser.parse_args()
print(args)
# ====================== prepare data and model faster rcnn ===============================
device = "cuda"
max_item = 100
train_data = qdataset("training", args.shape, args.data_train, max_item)
train_load = torch.utils.data.DataLoader(dataset=train_data,
                                         batch_size=args.batch_size,
                                         shuffle=True)

test_data = qdataset("testing", args.shape, args.data_test, max_item)
test_load = torch.utils.data.DataLoader(dataset=test_data,
                                         batch_size=1,
                                         shuffle=True)
# print(len(test_load))
faster_rcnn = resnet_fpn(classes=10)
# faster_rcnn = detnet_fpn(classes=10)
faster_rcnn.create_architecture()
faster_rcnn = faster_rcnn.to(device)
# ==========================================================================================

def optimizer_loader(weight_decay):
    params = []
    lr = args.lr_max
    for key, value in dict(faster_rcnn.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]

    if args.opt == "adam":
        return torch.optim.Adam(params)
    elif args.opt == "sgd":
        return torch.optim.SGD(params, momentum=0.9)

optimizer = optimizer_loader(0.0005)

def train(epoch):
    print("Training epoch:{}".format(epoch))

    train_loss = 0
    now = time.time()
    for idx, (data, info, gt, num) in enumerate(train_load):
        data, info, gt, num = data.to(device), info.to(device), gt.to(device), num.to(device)
        faster_rcnn.zero_grad()

        rois, objectiness, bbox_location, rois_label, loss = faster_rcnn(data, info, gt, num)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        clipping(faster_rcnn, 10.0)
        optimizer.step()

        if (idx+1) % args.display == 0:
            loss_now = (train_loss / (idx + 1))
            time_now = time.time() - now
            total_data = len(train_data)
            done_data = idx * args.batch_size
            now = time.time()
            print("Data: {} / {} Loss:{:.7f}  Time:{:.2f} s".format(done_data, total_data, loss_now, time_now))

def show_image(img, myclass, final_location, thresh=0.5):
    # show_transforms = transforms.ToPILImage(mode="RGB")

    for i in range(final_location.shape[0]):
        bbox = tuple(int(np.round(x)) for x in final_location[i, :4])
        score = final_location[i, -1]
        if score > thresh:
            cv2.rectangle(img, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(img, str(myclass), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)

    return img
    # return im
    # img = show_transforms(img)
    # img = np.array(img)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # pass

def test(epoch):

    test_thresh = 0.05
    path_result_image = args.data_test+"result/"
    for idx, (data, info, gt, num) in tqdm(enumerate(test_load)):
        data, info, gt, num = data.to(device), info.to(device), gt.to(device), num.to(device)
        rois, objectiness, bbox_location, rois_label, loss = faster_rcnn(data, info, gt, num)
        objectiness = objectiness.data
        boxes = rois.data[:, :, 1:5]

        boxes = boxes.cpu()
        _ = torch.from_numpy(np.tile(boxes, (1, objectiness.shape[1])))
        pred_boxes = _.cuda()

        objectiness = objectiness.squeeze()
        pred_boxes = pred_boxes.squeeze()

        data1 = data
        img = data1.squeeze()
        img = img.cpu().permute(1, 2, 0).numpy()
        img = img.astype(np.uint8)

        for j in range(1, args.z_class):
            index = torch.nonzero(objectiness[:, j] > test_thresh).view(-1)
            # if there is det
            if index.numel() > 0:
                cls_scores = objectiness[:, j][index]
                _, order = torch.sort(cls_scores, 0, True)
                # bp()
                cls_boxes = pred_boxes[index][:, j * 4:(j + 1) * 4]
                final_location = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                final_location = final_location[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], 0.3)
                final_location = final_location[keep.view(-1).long()]
                img = show_image(img, j, final_location.cpu().numpy(),thresh=0.5)

        link_img = path_result_image+str(idx)+".jpg"
        cv2.imwrite(link_img, img)
            # Limit to max_per_image detections *over all classes*

def main():
    print("================================================================")
    print("Training batch size:", args.batch_size)

    # bp()
    args.epochs = 1
    for epoch in range(args.epochs):
        faster_rcnn.train()
        train(epoch)
        faster_rcnn.eval()
        test(epoch)

if __name__ == "__main__":
    print(">>> Run <<<")
    main()
    print(">>> All done <<<")
