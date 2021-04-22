import numpy as np
import torch
from torchvision.datasets.coco import CocoDetection
from torch.utils.data import DataLoader
from torchvision import transforms
from ssd import SSD300
from utils import MultiBoxLoss as Loss
from utils import PriorBox
from coco import coco_cfg as cfg


def build_predictions(ploc, plabel):
    '''
    Input:
      ploc: [batch_num, 4, 8732]
      plabel: [batch_num, 81, 8732]
    
    Output:
      predictions: A tuple containing loc preds, conf preds and prior boxes from SSD net
        loc_preds: [batch_num, 8732, 4]
        conf_preds: [batch_num, 8732, 81]
        prior: [8732, 4]
    '''
    loc_preds = ploc.permute(0, 2, 1)
    conf_preds = plabel.permute(0, 2, 1)
    prior = torch.autograd.Variable(PriorBox(cfg).forward(), volatile=True)
    return (loc_preds, conf_preds, prior)


def build_targets(bbox, label):
    '''
    Input:
      bbox: tuple of num_bbox, tuple of 4, tensor [batch_num]
      label: tuple of num_box, tensor [batch_num]

    Output:
      targets: [batch_size,num_objs,5] (last idx is the label)
    '''
    _bbox = torch.stack([torch.stack(b) for b in bbox]).permute(2, 0, 1).float()
    _label = torch.stack(label).permute(1, 0).unsqueeze(2).float()
    return torch.cat([_bbox, _label], 2)


if __name__ == "__main__":

    train_image = r'../data/coco/images/train2017'
    train_info = r'../data/coco/annotations/instances_train2017.json'
    val_image = r'../data/coco/images/val2017'
    val_info = r'../data/coco/annotations/instances_val2017.json'
    batch_size = 2
    total_epoch = 65

    # ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math='fp32')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("# using {}, total GPU number is {}".format(device, torch.cuda.device_count()), flush=True)

    model = SSD300().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.35,verbose=True,min_lr=0.000001,patience=300)

    loss_func = Loss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, torch.cuda.is_available())
    loss_func.to(device)

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_dataset = CocoDetection(root=train_image, annFile=train_info, transform=transform)
    # train_dataset = COCODetection(train_image, train_info, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_dataset = CocoDetection(root=val_image, annFile=val_info, transform=transform)
    # val_dataset = COCODetection(val_image, val_info, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=0, drop_last=True)
    print('Number of train samples: {}, number of validation samples: {}'.format(len(train_dataset), len(val_dataset)))

    for epoch_idx in range(0, total_epoch):
        train_acc, test_acc = 0, 0
        for batch_idx, sample in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            img = sample[0].to(device)
            bbox = [s['bbox'] for s in sample[1]]
            label = [s['category_id'] for s in sample[1]]
            if len(bbox) == 0:
                print("No objetcts in batch")
                continue

            ploc, plabel = model(img)
            predictions = build_predictions(ploc, plabel)
            targets = build_targets(bbox, label)
            loss_l, loss_c = loss_func(predictions, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            print("loss: {}".format(loss.cpu().item()))
        # for batch_idx, sample in enumerate(test_loader):
        #     model.eval()
        #     data, label = sample[b'data'].float().to(device), sample[b'label'].long().to(device)
        #     pred = model(data)
        #     loss = loss_func(pred, label)
        #     _, pred_label = torch.max(pred, 1)
        #     test_acc += (pred_label == label).sum().item()
        # print("epoch {},  train acc: {:.3f}, test acc: {:.3f}".format(epoch_idx, train_acc / len(train_loader), test_acc / len(test_loader)))