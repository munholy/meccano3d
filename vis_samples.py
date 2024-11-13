import glob
import random
from PIL import Image
import os

def sample_and_save_grid(image_path, output_path, grid_size=(4, 4), sample_size=16, resolution=(100, 100)):
    # 특정 경로 아래의 모든 .jpg 파일을 재귀적으로 찾음
    images = glob.glob(os.path.join(image_path, '**', '*.jpg'), recursive=True)
    
    # 이미지가 충분히 있는지 확인
    if len(images) < sample_size:
        raise ValueError(f"경로에 최소 {sample_size}개의 jpg 파일이 필요합니다.")
    
    # 16개의 이미지를 무작위로 샘플링
    sampled_images = random.sample(images, sample_size)
    
    # 각 이미지를 해상도에 맞게 조정
    resized_images = [Image.open(img).resize(resolution) for img in sampled_images]
    
    # 격자 이미지 크기 설정
    grid_width, grid_height = grid_size
    single_width, single_height = resolution
    grid_image = Image.new('RGB', (grid_width * single_width, grid_height * single_height))
    
    # 격자에 이미지 배치
    for i, img in enumerate(resized_images):
        x = (i % grid_width) * single_width
        y = (i // grid_width) * single_height
        grid_image.paste(img, (x, y))
    
    # 격자 이미지 저장
    grid_image.save(output_path)
    print(f"이미지를 {output_path}에 저장했습니다.")

# 사용 예시
image_path = 'output/bop_data/meccano3d/train_pbr'  # 이미지 파일이 있는 경로
output_path = 'sample_grid.jpg'      # 저장할 출력 파일 이름
sample_and_save_grid(image_path, output_path, resolution=(1024, 1024))