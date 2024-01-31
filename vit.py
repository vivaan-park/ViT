import torch
import torch.nn as nn
from einops import rearrange

from transformer import Transformer
from dataloader import *
from params import *


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth,
                 heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be ' \
                                             'divisible by the patch size '
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.transformer(x, mask)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


def train(device, model, loader, criterion, optimizer):
    model.train()

    running_loss = .0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()

    epoch_loss = running_loss / len(loader)
    accuracy = 100 * correct / total

    return epoch_loss, accuracy


def validate(device, model, loader, criterion):
    model.eval()

    running_loss = .0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()

        epoch_loss = running_loss / len(loader)
        accuracy = 100 * correct / total

        return epoch_loss, accuracy


def main():
    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'>>> Using {device} device <<<')

    print('>>> Loading datasets <<<')
    train_dataset, val_dataset = load_data()
    train_loader = dataloader(train_dataset, batch_size)
    val_loader = dataloader(val_dataset, batch_size)

    print('>>> Building model <<<')
    vit = ViT(image_size=image_size,
              patch_size=patch_size,
              num_classes=num_classes,
              channels=channels,
              dim=dim,
              depth=depth,
              heads=heads,
              mlp_dim=mlp_dim)
    vit.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vit.parameters(), lr=learning_rate)

    for e in range(epoch):
        train_loss, train_acc = train(device, vit, train_loader, criterion, optimizer)
        valid_loss, valid_acc = validate(device, vit, val_loader, criterion)

        print(f'> train loss: {train_loss:.4f}, train accuracy: {train_acc:.2f}% | '
              f'validation loss: {valid_loss:.4f}, validation accuracy: {valid_acc:.2f}%')


if __name__ == '__main__':
    main()
