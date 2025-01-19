from datasets import load_dataset
import matplotlib.pyplot as plt
import torch
from transformers import CLIPProcessor, CLIPModel
from gpu import get_device

# 7번 GPU 설정
device = get_device(6)

image_dataset = load_dataset("eBoreal/food-500-enriched")
image = image_dataset["train"]["image"]

# 모델 및 프로세서 불러오기
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 학습된 가중치 로드
model.load_state_dict(torch.load("best_model.pt"))
model.eval()  # 평가 모드로 설정


def find_most_similar_image_from_dataset(text, images, model, processor):
    """
    텍스트와 가장 유사도가 높은 이미지를 찾는 함수.
    images는 PIL.Image 객체 리스트여야 합니다.
    """
    if not images:
        raise ValueError("The images list is empty. Provide valid PIL.Image objects.")

    # 텍스트 임베딩 계산
    text_inputs = processor(
        text=[text], return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )  # 정규화

    # 이미지 임베딩 계산
    image_features = []
    valid_images = []  # 처리 성공한 이미지 저장
    for idx, image in enumerate(images):
        try:
            # PIL.Image 객체를 사용하여 처리
            image_inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                img_features = model.get_image_features(**image_inputs)
                img_features = img_features / img_features.norm(
                    dim=-1, keepdim=True
                )  # 정규화
                image_features.append(img_features)
                valid_images.append(image)  # 처리 성공한 이미지 저장
        except Exception as e:
            print(f"Error processing image at index {idx}: {e}")
            continue

    # 예외 처리: 유효한 이미지가 없는 경우
    if not image_features:
        raise ValueError(
            "No valid images were processed. Check the image data and formats."
        )

    # 텍스트와 이미지 간의 유사도 계산
    similarities = [
        torch.cosine_similarity(text_features, img_feat).item()
        for img_feat in image_features
    ]

    # 가장 유사도가 높은 이미지와 유사도 값 찾기
    max_similarity = max(similarities)  # 가장 높은 유사도 값
    max_index = similarities.index(
        max_similarity
    )  # 해당 유사도 값을 가지는 이미지 인덱스
    most_similar_image = valid_images[max_index]  # 해당 이미지를 반환

    return most_similar_image, max_similarity


text_query = input("어떤 음식의 이미지를 찾으시나요?")

# 가장 유사한 이미지 찾기
try:
    most_similar_image, max_similarity = find_most_similar_image_from_dataset(
        text_query, image, model, processor
    )
    print(f"The most similar image is at index: {image.index(most_similar_image)}")
    print(f"Similarity: {max_similarity}")

    # 이미지 시각화
    plt.imshow(most_similar_image)
    plt.axis("off")
    plt.show()

except ValueError as e:
    print(f"Error: {e}")
