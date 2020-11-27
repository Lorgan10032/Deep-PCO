import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch

from FirstDataset import FirstDataset


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    full_dataset = FirstDataset(img_dir='D:\\2020\\PCO项目\\严院长病人前节PCO\\可用图像',
                                annotation_dir='D:\\2020\\PCO项目\\严院长病人前节PCO\\label', transforms=transforms.ToTensor())
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # define training and validation data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=30, shuffle=True, num_workers=4,
                                               collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=30, shuffle=False, num_workers=4,
                                              collate_fn=collate_fn)

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # freeze all the network except the final layer
    for param in model.parameters():
        param.requires_grad = False
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (lens) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        # train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        model.train()
        for images, targets in train_loader:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            # update the learning rate
            lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, test_loader, device=device)
