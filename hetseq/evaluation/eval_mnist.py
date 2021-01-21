import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, target, eval=False):
        # print('shape', x.shape, target.shape)
        # print(target)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        loss = F.nll_loss(output, target)
        return loss if not eval else output, loss
        # return loss

def main(args):
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    dataset2 = datasets.MNIST(args.mnist_dir, train=False, download=True,
                           transform=transform)

    device = torch.device("cuda")
    checkpoint = torch.load(args.model_ckpt)

    model = MNISTNet()
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    kwargs = {'batch_size': 64}
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)


    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data, target, eval=True)
            output = outputs[0]
            loss = outputs[1]
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='evaluate trained mnist model',
        allow_abbrev=False,
    )

    parser.add_argument(
        '--model_ckpt',
        type=str,
        default='/scratch365/yding4/hetseq/CRC_RUN_FILE/new_test/node1gpu4/checkpoint_last.pt',
        help='Specify the input AIDA directory',
    )

    parser.add_argument(
        '--mnist_dir',
        type=str,
        default='/scratch365/yding4/mnist/MNIST/processed',
        help='Specify the input AIDA directory',
    )


    args = parser.parse_args()
    main(args)
