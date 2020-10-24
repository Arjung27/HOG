import csv
import os
import argparse

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import vgg
from train import AverageMeter, accuracy

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--cleanmodel', action='store_true')

args = parser.parse_args()


def test(dataloader, model):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    for i, (input, target) in enumerate(dataloader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input)

        prec1, prec5 = accuracy(output, target, topk=(1,5))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        # if i % configs.TRAIN.print_freq == 0:
        #     print('Test: [{0}/{1}]\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #            i, len(test_iter), top1=top1))
        #     sys.stdout.flush() 
    return top1, top5

if args.cleanmodel:
    model = models.vgg11(pretrained=True)
    model = torch.nn.DataParallel(model).cuda()
    test_log = 'cleanVgg_11' + '.csv'
else:
    model = vgg.vgg11_H1(pretrained=False)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load('exp_vgg11_H1_0.01/checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    test_log = 'exp_vgg11_H1_0.01/checkpoint.pth.tar' + '.csv'
model.eval()

result = open(test_log, "wt", newline="")
cw = csv.writer(result)
cw.writerow([test_log])
cw.writerow(["test", "error top1", "error top5", "normalized error"])
    
####################### clean imagenet #######################
testdir = os.path.join('/fs/cml-datasets/ImageNet/ILSVRC2012', 'val')
test_iter = torch.utils.data.DataLoader(
            datasets.ImageFolder(testdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])), batch_size=64, shuffle=False,
            num_workers=4, pin_memory=True)
result, top5 = test(test_iter, model)
print(' Clean Prec@1 {result.avg:.3f}\n'.format(result=result))
cw.writerow(["\t\t\t cleantest \t\t\t"])
cw.writerow(["cleantest", str(100 - result.avg.item()), str(100 - top5.avg.item()), "--"])


testdir = os.path.join('/cmlscratch/manlis/data/Stylized-ImageNet', 'val')
test_iter = torch.utils.data.DataLoader(
            datasets.ImageFolder(testdir, transforms.Compose([
                #transforms.Resize(configs.DATA.img_size),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])), batch_size=64, shuffle=False,
            num_workers=4, pin_memory=True)
result, top5 = test(test_iter, model)
print(' stylized-imagenet: Prec@1 {result.avg:.3f}\n'.format(result=result))
cw.writerow(["\t\t\t stylized imagenet \t\t\t"])
cw.writerow(["stylized imagenet", str(100 - result.avg.item()), str(100 - top5.avg.item()), "--"])

####################### insta imagenet #######################
testdir = '/cmlscratch/manlis/data/instagram_val'
filters = [f.path for f in os.scandir(testdir) if f.is_dir()]
total_err_1 = 0
total_err_5 = 0
for filter in filters:
    subdir = os.path.join(testdir, filter)
    test_iter = torch.utils.data.DataLoader(
                    datasets.ImageFolder(subdir, transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])), batch_size=64, shuffle=False,
                    num_workers=4, pin_memory=True)
    result, top5 = test(test_iter, model)
    cw.writerow([filter, str(100 - result.avg.item()), str(100 - top5.avg.item()), "--"])
    total_err_1 += (100.0 - result.avg.item())
    total_err_5 += (100.0 - top5.avg.item())
total_err_1 = total_err_1 / (len(filters))
total_err_5 = total_err_5 / (len(filters))
print('mean Error over 20 filter types: @1 {:.3f}/ @5 {:.3f}'.format(total_err_1, total_err_5))
cw.writerow(["\t\t\t insta-imagenet \t\t\t"])
cw.writerow(["mCE", "--", "--", str(total_err_1), '/', str(total_err_5)])


