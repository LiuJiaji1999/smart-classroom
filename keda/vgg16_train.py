# from sklearn import datasets
import torch
import torchvision
from torchvision import transforms
import os
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# from utils import save_checkpoint
import utils
from torch.autograd import Variable

def dataload(trainData, testData):
    # 训练数据
    train_data = torchvision.datasets.ImageFolder(trainData, transform=transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]))
    train_loader = DataLoader(train_data, batch_size=20, shuffle=True)

    # 测试数据
    test_data = torchvision.datasets.ImageFolder(testData, transform=transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]))
    test_loader = DataLoader(test_data, batch_size=20, shuffle=True)
    return train_data, test_data, train_loader, test_loader

def train(model, trainData, testData):
    criterion = torch.nn.CrossEntropyLoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    train_data, test_data, train_loader, test_loader = dataload(trainData, testData)

    log = []
    # 启动训练
    epoches = 240
    for epoch in range(epoches):
        train_loss = 0.
        train_acc = 0.
        for step, data in enumerate(train_loader):
            batch_x, batch_y = data
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()  # GPU
            out = model(batch_x)
            loss = criterion(out, batch_y)
            train_loss += loss.item()
            # pred is the expect class
            # batch_y is the true label
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print('Epoch: ', epoch, 'Step', step,
                      'Train_loss: ', train_loss / ((step + 1) * 20), 'Train acc: ', train_acc / ((step + 1) * 20))

        print('Epoch: ', epoch, 'Train_loss: ', train_loss / len(train_data), 'Train acc: ',
              train_acc / len(train_data))

        # 保存训练过程数据
        info = dict()
        info['Epoch'] = epoch
        info['Train_loss'] = train_loss / len(train_data)
        info['Train_acc'] = train_acc / len(train_data)
        log.append(info)

    # 模型保存
    model_without_ddp = model
    os.chdir('/data/liujiaji/action/vgg16log/')
    dir_name = time.strftime('%m-%d-%Hh%Mm')
    os.mkdir(dir_name)
    # utils.save_checkpoint({
    torch.save({
        'model': model_without_ddp.state_dict()},
        os.path.join(dir_name, 'model.pth'))
    draw(log, dir_name)
    # draw(log)
    model.eval()

    os.chdir('../')
    eval_loss = 0
    eval_acc = 0
    for step, data in enumerate(test_loader):
        batch_x, batch_y = data
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        eval_loss += loss.item()
        # pred is the expect class
        # batch_y is the true label
        pred = torch.max(out, 1)[1]
        test_correct = (pred == batch_y).sum()
        eval_acc += test_correct.item()
    
    print('Test_loss: ', eval_loss / len(test_data), 'Test acc: ', eval_acc / len(test_data))


def draw(logs: list,dir_name):
    plt.figure()
    epoch = []
    loss = []
    acc = []
    for log_ in logs:
        epoch.append(log_['Epoch'])
        loss.append(log_['Train_loss'])
        acc.append(log_['Train_acc'])
    plt.plot(epoch, loss, 'r-', label='loss')
    plt.plot(epoch, acc, 'b-', label='accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(dir_name, 'trainProcess.jpg'))
    plt.show()

if __name__ == '__main__':
    num_cls = 8
    model = torchvision.models.vgg16(pretrained=True) # 加载torch原本的vgg16模型，设置pretrained=True，即使用预训练模型
    num_fc = model.classifier[6].in_features # 获取最后一层的输入维度
    model.classifier[6] = torch.nn.Linear(num_fc, num_cls)# 修改最后一层的输出维度，即分类数
    # 对于模型的每个权重，使其不进行反向传播，即固定参数
    for param in model.parameters():
        param.requires_grad = False
    # 将分类器的最后层输出维度换成了num_cls，这一层需要重新学习
    for param in model.classifier[6].parameters():
        param.requires_grad = True
    print(model)
    # model.cuda()

    # trainData = '/data/liujiaji/action/kedaclassroom/vggimg/train'
    # testData = '/data/liujiaji/action/kedaclassroom/vggimg/test'

    # train(model,trainData,testData)


