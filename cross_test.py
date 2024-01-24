import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch import nn
from torch.backends import cudnn
from Ada.dataset.raf import Dataset_RAF
from Ada.models.backbone import ResNet_18
from main import validate


def cross_over_test(model_path, test_path, test_label_path, use_cuda=False):
    '''
    跨数据集测试
    :return:
    '''
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    np.random.seed(5)
    torch.manual_seed(5)
    torch.cuda.manual_seed_all(5)

    criterion = nn.CrossEntropyLoss(reduction='none')

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_set = Dataset_RAF(test_path, test_label_path, transform=transform)
    test_loader = data.DataLoader(test_set, shuffle=False, batch_size=64, num_workers=0)

    # 加载模型
    model = ResNet_18(num_classes=7, checkpoint_path='Ada/models/resnet18_msceleb.pth')
    for param in model.parameters():
        param.detach_()
    state_dict = torch.load(model_path)['ema_state_dict']
    model = nn.DataParallel(model)
    cudnn.benchmark = True
    model.load_state_dict(state_dict, strict=True)

    model = model.eval()
    _, test_acc, _, _ = validate(test_loader, model, criterion, epoch=None, use_cuda=use_cuda, mode='Test Stats')
    # print(test_acc)
    return test_acc
