import torch
import argparse
from datasets import qdataset
from tqdm import tqdm
from model.vgg_backbone import vgg_backbone
from tools.dl import clip_gradient as clipping
import time

# hello world
# ==========================================================================================
parser = argparse.ArgumentParser(description='faster-rcnn')
parser.add_argument('--batch', dest='batch_size',
                    help='batch_size',
                    default=3, type=int)
parser.add_argument('--data_train', dest='data_train',
                    help='data_train',
                    default="data/train/", type=str)

parser.add_argument('--data_test', dest='data_test',
                    help='data_test',
                    default="data/test/", type=str)

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


args = parser.parse_args()
print(args)
# ====================== prepare data and model faster rcnn ===============================
device = "cuda"
train_data = qdataset("training", args.shape, args.data_train, 100)
train_load = torch.utils.data.DataLoader(dataset=train_data,
                                         batch_size=args.batch_size,
                                         shuffle=True)

test_data = qdataset("testing", args.shape, args.data_test, 100)
test_load = torch.utils.data.DataLoader(dataset=train_data,
                                         batch_size=args.batch_size,
                                         shuffle=True)

faster_rcnn = vgg_backbone(classes=10, pretrained=True)
faster_rcnn.build()
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

        rois, class_prob, bbox_pred, rois_label, loss = faster_rcnn(data, info, gt, num)
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
            print("Data: {} / {} Loss:{:.7f}  Time:{:.2f} s".format(done_data,
                                                                    total_data, loss_now, time_now))

def test(epoch):
    # print()
    pass
    # for idx, (data, info, gt, num) in enumerate(test_load):
    #     pass

def main():
    print("================================================================")
    print("Training batch size:", args.batch_size)

    for epoch in range(args.epochs):
        faster_rcnn.train()
        train(epoch)
        # faster_rcnn.eval()
        # test(epoch)

if __name__ == "__main__":
    print(">>> Run <<<")
    main()
