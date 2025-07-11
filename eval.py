import torch
from utils import *
import resnet_cifar

def eval(trained_model):
    _, eval_loader = load_data()

    correct = 0
    total = 0

    for i, data in enumerate(eval_loader):
        images, labels = data
        outputs = trained_model(images)
        _, prediction = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (labels == prediction).sum()
    
    print("Accuracy of test_loader: %.3f %%" % (correct/total*100.0))


if __name__ == "__main__":
    trained_model = resnet_cifar.__dict__["resnet56_build"]()
    trained_model.load_state_dict(torch.load("resnet56.pth"))
    eval(trained_model)




