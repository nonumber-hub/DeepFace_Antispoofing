import torch
from torchvision import models
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import torch.optim as optim
from torch import tensor

# JSON 파일을 읽고 이미지를 처리하는 함수
def load_labels(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)  # JSON 파일을 로드합니다.
    return data

# 커스텀 데이터셋 클래스 정의
class FaceDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        self.data = load_labels(json_file)  # JSON 파일에서 데이터 로드
        self.root_dir = root_dir  # 데이터의 기본 경로
        self.transform = transform  # 이미지 전처리

        # 이미지 경로와 레이블을 리스트로 저장
        self.image_paths = list(self.data.keys())  # 이미지 경로 리스트
        self.labels = list(self.data.values())  # 레이블 리스트

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])  # 이미지 경로
        if not os.path.exists(img_path):  # 경로가 존재하는지 체크
            print(f"Warning: Image {img_path} not found.")
            return None, None  # 파일이 없으면 None 반환

        image = Image.open(img_path).convert("RGB")  # 이미지를 열고 RGB로 변환
        label = self.labels[idx]  # 레이블

        if self.transform:
            image = self.transform(image)  # 변환 적용

        # 레이블을 텐서로 변환하고, CrossEntropyLoss가 요구하는 형태로 1D 텐서로 변환
        label = tensor(label, dtype=torch.long)  # 정수형 텐서로 변환

        return image, label

# 데이터셋 경로 설정
train_json_path = 'metas/protocol2/test_on_high_quality_device/train_label.json'
test_json_path = 'metas/protocol2/test_on_high_quality_device/test_label.json'

# 훈련 데이터셋 로딩
train_data = FaceDataset(json_file=train_json_path, root_dir='', transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))

# 테스트 데이터셋 로딩
test_data = FaceDataset(json_file=test_json_path, root_dir='', transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))

# 데이터로더 설정
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 사전 학습된 모델 불러오기
model = models.resnet50(weights="IMAGENET1K_V1")

# 출력층을 이진 분류에 맞게 변경 (2개의 클래스: live, spoof)
model.fc = nn.Linear(model.fc.in_features, 2)

# 모델을 GPU로 이동 (가능한 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 손실 함수 (이진 크로스 엔트로피)
criterion = nn.CrossEntropyLoss()

# 옵티마이저 (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 학습 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            if inputs is None or labels is None:
                continue  # None 값이 있으면 건너뜁니다.

            inputs, labels = inputs.to(device), labels.to(device)
            
            # 기울기 초기화
            optimizer.zero_grad()

            # 순전파
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 역전파
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 에폭 종료 후 결과 출력
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# 학습 실행
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# 평가 함수
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            if inputs is None or labels is None:
                continue  # None 값이 있으면 건너뜁니다.
            
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# 평가 실행
evaluate_model(model, test_loader)

# 모델 저장
torch.save(model.state_dict(), 'face_spoof_model.pth')

# 모델 로드 (CPU에서 로드하도록 명시)
model.load_state_dict(torch.load('face_spoof_model.pth', map_location=device))
model.eval()
