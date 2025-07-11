import torch
import torchvision.transforms as transforms
from resnet_cifar import *
from utils import *
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的是：{device}")
    if device.type == 'cuda':
        print("CUDA device name:", torch.cuda.get_device_name(0))

    train_loader, _ = load_data()
    net = resnet56_build(weight=None, process=True)
    net = net.to(device)
    #print(net)

    f = open("resnet56_log.txt", mode='wt', encoding='utf-8')
    f.write('resnet56_cifar100\n')
    writer = SummaryWriter(comment='resnet56_cifar')

    criterion, optimizer = Optimizer(net)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)

    for epoch in range(200):
        net.train()
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            def compute_acc(output, label):
                prediction = torch.softmax(output, dim=1)
                return (prediction.argmax(dim=1)==label).type(torch.float).mean()
            
            acc = compute_acc(outputs, labels)

            loss.backward()
            optimizer.step()

            running_acc += acc.item()
            running_loss += loss.item()

            if i%25 == 24:
                print("[%d %5d] loss: %.3f acc: %.3f" % (epoch+1, i/25+1, running_loss/25, running_acc/25))
                f.write("[%d %5d] loss: %.3f acc: %.3f" % (epoch+1, i/25+1, running_loss/25, running_acc/25))
                writer.add_scalar('loss/train', running_loss/25, epoch*len(train_loader)+i)
                writer.add_scalar('accuracy', running_acc/25, epoch*len(train_loader)+i)

                running_acc = 0.0
                running_loss = 0.0
        scheduler.step()

    writer.close()
    f.close()
    print("Finished Training")
    PATH = "./resnet56.pth"
    torch.save(net.state_dict(), PATH)    

