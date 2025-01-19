import os
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    BertTokenizer,
    ViTFeatureExtractor,
    CLIPModel,
    CLIPProcessor,
    TrainingArguments,
    Trainer,
    logging,
)
import sys

sys.path.append("/home/ai20201051")
from gpu import get_device


# 환경 변수 설정
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# Configurations
class CFG:
    max_text_tokens_length = 77
    text_backbone = "bert-base-uncased"
    image_backbone = "google/vit-base-patch16-224"
    device = get_device(6)  # 7번 GPU 설정
    batch_size = 16
    max_epochs = 15
    patience = 3
    factor = 0.1


# 변환 정의: 이미지를 텐서로 변환하고 정규화
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # ViT 모델용 정규화 값
    ]
)


# PIL 이미지를 텐서로 변환하는 함수
def pil_to_tensor(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform = transforms.ToTensor()  # PIL 이미지를 텐서로 변환하는 transform
    return transform(image)


# Custom collate_fn 정의
def collate_fn(batch):
    # 텍스트가 없는 경우를 대비하여 예외 처리 추가
    try:
        batch_texts = [item["text"] for item in batch if "text" in item]
        batch_images = [item["image"] for item in batch if "image" in item]

        # 텍스트가 없으면 빈 문자열로 처리
        if not batch_texts:
            batch_texts = [""] * len(batch_images)

        # Processor 처리
        inputs = processor(
            text=batch_texts, images=batch_images, return_tensors="pt", padding=True
        )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
        }

    except KeyError as e:
        print(f"KeyError: {e}")
        return None


# CustomDataset 정의
class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])  # 이미지를 열기
        if self.transform:
            image = self.transform(image)  # 변환 적용
        return image


# Helper Functions
def preprocess_data(batch, processor):
    try:
        # 텍스트 데이터가 없는 경우 빈 문자열을 넣음
        texts = batch.get(
            "text", [""] * len(batch["image"])
        )  # get()을 사용하여 기본값 지정
        images = batch["image"]

        # PIL 이미지를 텐서로 변환
        images = [pil_to_tensor(img) for img in images]

        # CLIPProcessor로 텍스트와 이미지를 처리
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # 반환할 데이터 딕셔너리
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
        }
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None


# Model Initialization
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# 모델을 GPU로 전송
model.to(CFG.device)


# Training Loop
def train_epoch(model, dataloader, optimizer):
    model.train()
    epoch_loss = 0.0
    for batch in tqdm(dataloader, desc="Training Epoch"):
        optimizer.zero_grad()

        # 데이터도 GPU로 전송
        input_ids = batch["input_ids"].to(CFG.device)
        attention_mask = batch["attention_mask"].to(CFG.device)
        pixel_values = batch["pixel_values"].to(CFG.device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_loss=True,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def validate_epoch(model, dataloader):
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation Epoch"):
            input_ids = batch["input_ids"].to(CFG.device)
            attention_mask = batch["attention_mask"].to(CFG.device)
            pixel_values = batch["pixel_values"].to(CFG.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_loss=True,
            )
            loss = outputs.loss
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


# Training Loop
def train_model(model, train_dataloader, val_dataloader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    best_loss = float("inf")
    for epoch in range(CFG.max_epochs):
        print(f"Epoch {epoch + 1}/{CFG.max_epochs}")
        train_loss = train_epoch(model, train_dataloader, optimizer)
        val_loss = validate_epoch(model, val_dataloader)
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print("New best model saved!")
        else:
            print("No improvement.")


processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
    max_length=CFG.max_text_tokens_length,
    truncation=True,
    padding="max_length",
)


def prepare_for_training(batch):
    """
    Processor를 사용해 데이터셋을 학습에 적합한 형태로 변환합니다.
    """

    # 텍스트 길이를 제한하여 전처리
    truncated_texts = [text[: CFG.max_text_tokens_length] for text in batch["text"]]

    inputs = processor(
        text=truncated_texts,
        images=batch["image"],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"],
    }


# dataset2 label
label_str = [
    "burger",
    "butter_naan",
    "chai",
    "chapati",
    "chole_bhature",
    "dal_makhani",
    "dhokla",
    "fried_rice",
    "idli",
    "jalebi",
    "kaathi_rolls",
    "kadai_paneer",
    "kulfi",
    "masala_dosa",
    "momos",
    "paani_puri",
    "pakode",
    "pav_bhaji",
    "pizza",
    "samosa",
]

# num_classes
num_classes = 20
# 클래스 별 샘플링 비율 지정
k = 30


# 각 클래스별로 5개의 데이터만 선택하는 함수 정의
def select_samples_per_class(dataset, num_classes, percent_per_class):
    # 클래스별로 데이터 인덱스를 저장할 딕셔너리 초기화
    class_samples = {i: [] for i in range(num_classes)}

    # 데이터셋 순회하면서 각 클래스별로 인덱스를 추가
    for idx, label in enumerate(dataset["label"]):
        class_idx = label  # 레이블이 클래스 번호라고 가정
        class_samples[class_idx].append(idx)
        # if len(class_samples[class_idx]) < samples_per_class:
        #     class_samples[class_idx].append(idx)

    # 선택된 인덱스 모으기
    # selected_indices = [idx for indices in class_samples.values() for idx in indices]
    selected_indices = []
    for class_idx, indices in class_samples.items():
        # 각 클래스에서 지정된 비율만큼 추출
        num_samples = max(
            1, int(len(indices) * percent_per_class / 100)
        )  # 최소 1개 보장
        selected_indices.extend(indices[:num_samples])  # 처음부터 num_samples만큼 선택

    # 선택된 인덱스를 사용하여 데이터셋 슬라이싱
    return dataset.select(selected_indices)


def add_text_feature(example):
    example["text"] = "A Photo of " + label_str[example["label"]]  # 주로 문장으로 학습
    return example


# Main Code
if __name__ == "__main__":
    # 데이터셋 로드
    # dataset1 = load_dataset("eBoreal/food-500-enriched")
    dataset1 = load_dataset("hectoritr/FoodDataset")
    dataset2 = load_dataset("ksuyash/food-dataset")

    # 데이터셋에 함수 적용
    dataset1 = dataset1.map(prepare_for_training, batched=True)
    dataset2 = dataset2.map(add_text_feature)
    dataset2 = select_samples_per_class(dataset2["train"], num_classes, k)

    # data1에서 필요한 feature만 유지
    dataset1 = dataset1.map(
        lambda x: {"image": x["image"], "text": x["text"]},
        remove_columns=dataset1["train"].column_names,
    )

    print(dataset1)
    print("-------------")
    print(dataset2)
    print("-------------")

    # data2에서 필요한 feature만 유지
    dataset2 = dataset2.map(
        lambda x: {"image": x["image"], "text": x["text"]},
        remove_columns=dataset2.column_names,
    )

    # Data preparation
    split_data1 = dataset1["train"].train_test_split(test_size=0.2)
    data1 = DatasetDict(
        {"train": split_data1["train"], "validation": split_data1["test"]}
    )

    split_data2 = dataset2.train_test_split(test_size=0.2)
    data2 = DatasetDict(
        {"train": split_data2["train"], "validation": split_data2["test"]}
    )

    print(data1)
    print("-----------------------")
    print(data2)
    print("-----------------------")

    # 두 데이터셋 병합
    merged_train_dataset = concatenate_datasets([data1["train"], data2["train"]])
    merged_val_dataset = concatenate_datasets(
        [data1["validation"], data2["validation"]]
    )

    train_dataset = merged_train_dataset.map(
        lambda batch: preprocess_data(batch, processor), batched=True
    )
    val_dataset = merged_train_dataset.map(
        lambda batch: preprocess_data(batch, processor), batched=True
    )

    # DataLoader에 collate_fn 추가
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=CFG.batch_size, collate_fn=collate_fn, drop_last=False
    )

    # Train the model
    train_model(model, train_dataloader, val_dataloader)
