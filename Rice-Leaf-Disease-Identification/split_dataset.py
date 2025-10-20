import os
import shutil
import random
from tqdm import tqdm

# === CẤU HÌNH ===
DATASET_DIR = 'dataset_original'  # Thư mục chứa dữ liệu gốc
OUTPUT_DIR = 'dataset'            # Thư mục đích sau khi chia
SPLIT_RATIOS = (0.7, 0.15, 0.15)  # train / val / test
MOVE_FILES = False                # Nếu True thì di chuyển (move), False thì copy
random.seed(42)                   # Giúp chia ngẫu nhiên nhưng có thể tái lập

# === KHỞI TẠO ===
train_dir = os.path.join(OUTPUT_DIR, 'train')
val_dir = os.path.join(OUTPUT_DIR, 'validation')
test_dir = os.path.join(OUTPUT_DIR, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Lấy danh sách class (thư mục con)
classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

print(f"📂 Phát hiện {len(classes)} lớp: {classes}")
print("🚀 Đang chia dữ liệu...")

for cls in classes:
    cls_dir = os.path.join(DATASET_DIR, cls)
    images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    random.shuffle(images)

    n_total = len(images)
    n_train = int(SPLIT_RATIOS[0] * n_total)
    n_val = int(SPLIT_RATIOS[1] * n_total)
    n_test = n_total - n_train - n_val

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    for subset, subset_imgs in zip(
        [train_dir, val_dir, test_dir],
        [train_imgs, val_imgs, test_imgs]
    ):
        subset_cls_dir = os.path.join(subset, cls)
        os.makedirs(subset_cls_dir, exist_ok=True)

        for img_name in tqdm(subset_imgs, desc=f"{cls} → {os.path.basename(subset)}", leave=False):
            src_path = os.path.join(cls_dir, img_name)
            dst_path = os.path.join(subset_cls_dir, img_name)
            if MOVE_FILES:
                shutil.move(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

print("\n✅ Hoàn thành chia dataset!")
print(f"Dữ liệu đã được lưu trong thư mục: {OUTPUT_DIR}")

# === THỐNG KÊ KẾT QUẢ ===
def count_images(directory):
    total = 0
    for cls in os.listdir(directory):
        path = os.path.join(directory, cls)
        if os.path.isdir(path):
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"{os.path.basename(directory):<12} | {cls:<25} : {count} ảnh")
            total += count
    print(f"Tổng cộng trong {os.path.basename(directory)}: {total} ảnh\n")

print("\n📊 Thống kê sau khi chia:")
count_images(train_dir)
count_images(val_dir)
count_images(test_dir)
