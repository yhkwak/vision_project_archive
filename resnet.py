import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import optim

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, inner_channels, stride=1, projection=None):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, inner_channels * self.expansion, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels * self.expansion),
        )

        self.projection = projection
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.residual(x)

        if self.projection is not None:
            skip_connection = self.projection(x)
        else:
            skip_connection = x

        out = self.relu(residual + skip_connection)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, inner_channels, stride=1, projection=None):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, inner_channels * self.expansion, 1, bias=False),
            nn.BatchNorm2d(inner_channels * self.expansion),
        )

        self.projection = projection
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.residual(x)

        if self.projection is not None:
            skip_connection = self.projection(x)
        else:
            skip_connection = x

        out = self.relu(residual + skip_connection)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_block_list, n_classes=1000):
        super().__init__()
        assert len(num_block_list) == 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.in_channels = 64
        self.stage1 = self.make_stage(block, 64, num_block_list[0], stride=1)
        self.stage2 = self.make_stage(block, 128, num_block_list[1], stride=2)
        self.stage3 = self.make_stage(block, 256, num_block_list[2], stride=2)
        self.stage4 = self.make_stage(block, 512, num_block_list[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, n_classes)

    def make_stage(self, block, inner_channels, num_blocks, stride=1):
        if stride != 1 or self.in_channels != inner_channels * block.expansion:
            projection = nn.Sequential(
                nn.Conv2d(self.in_channels, inner_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(inner_channels * block.expansion),
            )
        else:
            projection = None

        layers = []
        for idx in range(num_blocks):
            if idx == 0:
                layers.append(block(self.in_channels, inner_channels, stride, projection))
                self.in_channels = inner_channels * block.expansion
            else:
                layers.append(block(self.in_channels, inner_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

def main():
    # 이미지 전처리 파이프라인 정의
    # 하이퍼파라미터 설정
    batch_size = 32
    learning_rate = 0.001
    epochs = 10
    n_classes = 2  # 데이터셋 클래스 수

    # 데이터 로딩 및 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 이미지 크기 조정
        transforms.ToTensor()
    ])

    train_dataset = ImageFolder(root='dataset/Training/01.raw_data', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # ResNet 모델 생성
    model = resnet50(n_classes=n_classes)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 루프
    for epoch in range(epochs):
        model.train()  # 모델을 학습 모드로 설정
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 손실 및 정확도 계산
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 에포크별 결과 출력
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total * 100
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    print("학습 완료!")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()  # Windows에서 필요할 수 있음
    main()

