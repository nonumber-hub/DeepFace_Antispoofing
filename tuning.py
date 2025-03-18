import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 사용자 정의 데이터셋 클래스 예시
class AntiSpoofingDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])  # 이미지 로드
        label = self.labels[idx]  # 라벨 (실제 또는 스푸핑)
        facial_area = self.get_facial_area(image)  # 얼굴 영역 정보 (선택적)

        # 변환 적용
        if self.transform:
            image = self.transform(image)

        return image, label, facial_area

    def get_facial_area(self, image):
        # 얼굴 탐지 방법을 추가하거나, 이미 bounding box가 있을 경우 이를 사용할 수 있습니다.
        pass

# 데이터셋 및 데이터로더 초기화
train_dataset = AntiSpoofingDataset(train_image_paths, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 옵티마이저 및 손실 함수 정의
optimizer = optim.Adam(list(first_model.parameters()) + list(second_model.parameters()), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

# 훈련 루프
for epoch in range(num_epochs):
    first_model.train()
    second_model.train()

    for images, labels, facial_area in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 두 모델에 대해 forward pass
        optimizer.zero_grad()
        first_output = first_model(images)
        second_output = second_model(images)

        # 결과를 결합하고 손실 계산
        combined_output = (first_output + second_output) / 2
        loss = criterion(combined_output, labels)

        # 역전파
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Fine-tuning 완료 후 모델 저장
torch.save(first_model.state_dict(), "fine_tuned_first_model.pth")
torch.save(second_model.state_dict(), "fine_tuned_second_model.pth")