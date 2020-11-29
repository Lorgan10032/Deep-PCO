import os
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from FirstDataset import FirstDataset
from PIL import Image, ImageDraw


def collate_fn(batch):
    return tuple(zip(*batch))


def cal_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h  # C∩G的面积
    iou = area / (s1 + s2 - area)
    return iou


if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    full_dataset = FirstDataset(img_dir='/home/shiyaohu/Data/pco_data/imgs',
                                annotation_dir='/home/shiyaohu/Data/pco_data/label', transforms=transforms.ToTensor())
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # define training and validation data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=15, shuffle=True, num_workers=4,
                                               collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=15, shuffle=False, num_workers=4,
                                              collate_fn=collate_fn)

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=1)
    # freeze all the network except the final layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.rpn.parameters():
        param.requires_grad = True
    # replace the classifier with a new one, that has num_classes which is user-defined
    num_classes = 2  # 1 class (lens) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # move model to the right device
    # if torch.cuda.device_count() > 1:
    #     print("Let's use ", torch.cuda.device_count(), "GPUs.")
    #     model = nn.DataParallel(model)
    model = model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 30

    for i, epoch in enumerate(range(num_epochs)):
        # train for one epoch, printing every 10 iterations
        # train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        print('epoch ' + str(i + 1) + ':')
        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets_cuda = []
            for target in targets:
                this_dict = {}
                for key in target:
                    this_dict[key] = target[key].cuda()
                targets_cuda.append(this_dict)
            model.train()
            loss_dict = model(images, targets_cuda)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            # update the learning rate
            # lr_scheduler.step()
            # evaluate on test set
        with torch.no_grad():
            model.eval()
            count = 0
            iou_sum = 0
            for images, targets in test_loader:
                images_cuda = list(img.to(device) for img in images)
                outputs = model(images_cuda)
                for i, output in enumerate(outputs):
                    count += 1
                    target_box = targets[i]['boxes'][0]
                    target_box = target_box.int().numpy().tolist()
                    if list(output['boxes']):
                        output_box = output['boxes'][0]
                        output_box = output_box.cpu().int().numpy().tolist()
                        iou = cal_iou(target_box, output_box)
                        iou_sum += iou
                        print(target_box, ' --- ', output_box, ' --- ', iou)
                    else:
                        iou = 0
                        iou_sum += iou
                        print(target_box, ' --- ', 'None', ' --- ', iou)
            print('mean iou:', iou_sum / count)
        print('-' * 50)

    with torch.no_grad():
        print('Generating img files...')
        model.eval()
        count = 0
        for images, targets in test_loader:
            images_cuda = list(img.to(device) for img in images)
            outputs = model(images_cuda)
            for i, output in enumerate(outputs):
                count += 1
                img = images[i]
                img = img.cpu().clone()
                img = transforms.ToPILImage()(img)
                draw = ImageDraw.Draw(img)
                target_box = targets[i]['boxes'][0]
                draw.rectangle(list(target_box), outline='red', width=4)
                if list(output['boxes']):
                    for box in output['boxes']:
                        draw.rectangle(list(box), outline='cyan', width=2)
                img.save('/home/shiyaohu/Data/pco_data/output/' + str(count).zfill(3) + ".jpg")

    print('Saving model...')
    torch.save(model.state_dict(), '/home/shiyaohu/Data/pco_data/model/mean_iou_{:.4f}.pkl'.format(iou_sum / count))
