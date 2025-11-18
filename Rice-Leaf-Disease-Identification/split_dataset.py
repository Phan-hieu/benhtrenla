import os
import shutil
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image, ImageEnhance

# === CẤU HÌNH ===
DATASET_DIR = "dataset_original"          # Thư mục chứa dữ liệu gốc
OUTPUT_DIR = "dataset"                    # Thư mục đích sau khi chia
SPLIT_RATIOS = (0.7, 0.15, 0.15)          # train / val / test
MOVE_FILES = False                        # True: di chuyển, False: copy
random.seed(42)
np.random.seed(42)

# === KHỞI TẠO THƯ MỤC ===
train_dir = os.path.join(OUTPUT_DIR, "train")
val_dir = os.path.join(OUTPUT_DIR, "validation")
test_dir = os.path.join(OUTPUT_DIR, "test")

for d in [train_dir, val_dir, test_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

# === LẤY DANH SÁCH CLASS ===
classes = [c for c in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, c))]
print(f"📂 Phát hiện {len(classes)} lớp: {classes}")

# === ĐẾM SỐ ẢNH MỖI LỚP ===
class_counts = {}
for cls in classes:
    cls_dir = os.path.join(DATASET_DIR, cls)
    imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    class_counts[cls] = len(imgs)

max_count = max(class_counts.values())
print("\n📊 Số lượng ảnh ban đầu:")
for cls, count in class_counts.items():
    print(f"   {cls:<25}: {count:>5} ảnh")
print(f"➡️  Mục tiêu cân bằng mỗi lớp: {max_count} ảnh\n")

# === HÀM TĂNG CƯỜNG ẢNH NHẸ (AUGMENTATION) ===
def augment_image(img_path, output_path):
    img = Image.open(img_path).convert("RGB")

    # Random flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Random brightness/contrast
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    img.save(output_path)

# === HÀM COPY/MOVE ===
def copy_or_move(src, dst):
    if MOVE_FILES:
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)

# === XỬ LÝ TỪNG LỚP ===
for cls in classes:
    print(f"\n🧩 Đang xử lý lớp: {cls}")
    cls_dir = os.path.join(DATASET_DIR, cls)
    images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"   Ảnh gốc: {len(images)}")

    # Oversample bằng cách copy + augment
    while len(images) < max_count:
        new_imgs = []
        for img_name in images:
            if len(images) + len(new_imgs) >= max_count:
                break
            src_img_path = os.path.join(cls_dir, img_name)
            new_img_name = f"aug_{len(images)+len(new_imgs)}_{img_name}"
            new_img_path = os.path.join(cls_dir, new_img_name)
            augment_image(src_img_path, new_img_path)
            new_imgs.append(new_img_name)
        images.extend(new_imgs)

    print(f"   Sau cân bằng: {len(images)} ảnh")

    # Chia tập train/val/test
    train_imgs, temp_imgs = train_test_split(images, train_size=SPLIT_RATIOS[0], random_state=42)
    val_ratio = SPLIT_RATIOS[1] / (SPLIT_RATIOS[1] + SPLIT_RATIOS[2])
    val_imgs, test_imgs = train_test_split(temp_imgs, train_size=val_ratio, random_state=42)

    for subset, subset_dir, img_list in [
        ("train", train_dir, train_imgs),
        ("validation", val_dir, val_imgs),
        ("test", test_dir, test_imgs)
    ]:
        dst_cls_dir = os.path.join(subset_dir, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)

        for img in tqdm(img_list, desc=f"   ➡️ {subset}", leave=False):
            copy_or_move(os.path.join(cls_dir, img), os.path.join(dst_cls_dir, img))

print("\n✅ Hoàn tất chia dữ liệu và cân bằng!\n")

# === THỐNG KÊ KẾT QUẢ ===
def count_images(directory):
    total = 0
    for cls in sorted(os.listdir(directory)):
        path = os.path.join(directory, cls)
        if os.path.isdir(path):
            n = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"{os.path.basename(directory):<12} | {cls:<25}: {n:>5} ảnh")
            total += n
    print(f"Tổng cộng trong {os.path.basename(directory)}: {total} ảnh\n")

count_images(train_dir)
count_images(val_dir)
count_images(test_dir)
