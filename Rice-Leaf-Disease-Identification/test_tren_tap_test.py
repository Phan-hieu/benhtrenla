import os
import numpy as np
import pandas as pd
import time
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import DenseNet121, DenseNet201
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# =========================================================
# ⚙️ CẤU HÌNH
# =========================================================
BATCH_SIZE = 16
TEST_DIR = 'dataset/test'
CLASS_NAMES = ['Bacterial leaf bright', 'Brown Spot', 'Healthy', 'Leaf Blast']
MODEL_SAVE_DIR = "models" 

# =========================================================
# 📦 HÀM TẠO TEST GENERATOR (LINH HOẠT)
# =========================================================
def create_test_generator(img_size):
    """Tạo test generator với kích thước ảnh cụ thể."""
    print(f"    ...Đang tạo test generator với kích thước {img_size}x{img_size}...")
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return test_generator

# =========================================================
# 📦 HÀM TẠO MÔ HÌNH LEGACY (TẢI TRỌNG SỐ)
# =========================================================
def build_densenet_model(architecture_name, weights_path, img_size):
    """Xây dựng lại kiến trúc và tải trọng số từ tệp .h5 gốc."""
    print(f"    ...Đang xây dựng kiến trúc {architecture_name}...")
    
    if architecture_name == "DenseNet201":
        base_model = DenseNet201(weights=None, include_top=False, input_shape=(img_size, img_size, 3))
    else:
        raise ValueError(f"Kiến trúc {architecture_name} không được hỗ trợ trong hàm này")

    x = base_model.output
    x = GlobalAveragePooling2D(name='global_average_pooling2d')(x)
    x = Dense(256, activation='relu', name='dense')(x)
    x = Dropout(0.3, name='dropout')(x)
    predictions = Dense(4, activation='softmax', name='dense_1')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    print(f"    ...Đang tải trọng số từ: {weights_path}...")
    model.load_weights(weights_path)
    
    return model

# =========================================================
# 🧠 DANH SÁCH 3 MÔ HÌNH SO SÁNH
# =========================================================
models_to_compare = {
    "DenseNet201": ("models/densenet201_best.h5", 224, "legacy"),
    "MobileNetV2": ("models/mobilenetv2_model.keras", 224, "modern"),
    "ResNet152V2": ("models/resnet152v2_rice_disease_best.keras", 256, "modern"),
}

# =========================================================
# 🚀 HÀM ĐÁNH GIÁ TỔNG HỢP
# =========================================================
def evaluate_all_models():
    print("=== 🔍 BẮT ĐẦU SO SÁNH CÁC MÔ HÌNH ===")
    results = []

    try:
        print("Đang lấy nhãn thực tế (true labels)...")
        temp_gen = create_test_generator(224) 
        true_classes = temp_gen.classes
        total_images = len(true_classes)
        print(f"    ...Đã tìm thấy {total_images} ảnh test.")
    except Exception as e:
        print(f"❌ LỖI NGHIÊM TRỌNG: Không thể tạo generator: {e}")
        return

    for name, (path, img_size, load_type) in models_to_compare.items():
        if not os.path.exists(path):
            print(f"\n❌ Bỏ qua: Không tìm thấy tệp '{path}'")
            continue

        print(f"\n🔍 Đang đánh giá: {name} (File: {path}, Size: {img_size}x{img_size})")
        
        try:
            if load_type == "legacy":
                model = build_densenet_model(name, path, img_size)
            else: 
                model = load_model(path)
        except Exception as e:
            print(f"    Lỗi khi tải mô hình '{path}': {e}")
            continue
            
        try:
            test_gen = create_test_generator(img_size)
        except Exception as e:
            print(f"    Lỗi khi tạo test generator: {e}")
            continue

        try:
            start_time = time.time()
            predictions = model.predict(test_gen, verbose=1)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time_per_image = (total_time / total_images) * 1000 
            predicted_classes = np.argmax(predictions, axis=1)
            
            acc = accuracy_score(true_classes, predicted_classes)
            report_dict = classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES, output_dict=True)

            print(f"✅ {name} Accuracy: {acc:.4f}")
            print(f"⏱️ {name} Tốc độ dự đoán: {total_time:.2f} giây ({avg_time_per_image:.2f} ms/ảnh)")
            print("\n--- Báo cáo phân loại chi tiết ---")
            print(classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES))
            print("----------------------------------\n")
            
            results.append((
                name, acc, 
                report_dict['macro avg']['precision'], 
                report_dict['macro avg']['recall'], 
                report_dict['macro avg']['f1-score'],
                avg_time_per_image
            ))

        except Exception as e:
            print(f"    Lỗi trong quá trình dự đoán: {e}")

    if not results:
        print("\nKhông có mô hình nào được đánh giá thành công.")
        return

    # In bảng so sánh cuối cùng
    df = pd.DataFrame(results, columns=[
        "Model", "Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)", "Avg. Time (ms/ảnh)"
    ])
    df = df.sort_values(by="Accuracy", ascending=False)
    
    print("\n\n==========================================")
    print("📋 KẾT QUẢ SO SÁNH TỔNG HỢP (3 MÔ HÌNH)")
    print("==========================================")
    print(df)
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    csv_path = os.path.join(MODEL_SAVE_DIR, "final_model_comparison_3_models.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nĐã lưu kết quả so sánh vào: {csv_path}")

    # === PHẦN TẠO BIỂU ĐỒ ===
    
    # Biểu đồ 1: Accuracy
    print("Đang tạo biểu đồ so sánh Accuracy...")
    plt.figure(figsize=(10, 6))
    plt.bar(df["Model"], df["Accuracy"], color="skyblue")
    plt.title("So sánh độ chính xác các mô hình")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.ylim(0.8, 1.0) 
    plt.grid(axis="y", linestyle='--', alpha=0.7)
    plt.tight_layout()
    png_path_acc = os.path.join(MODEL_SAVE_DIR, "model_comparison_accuracy.png")
    plt.savefig(png_path_acc)
    print(f"Đã lưu biểu đồ Accuracy vào: {png_path_acc}")
    try:
        plt.show() # CỐ GẮNG HIỂN THỊ BIỂU ĐỒ
    except Exception as e:
        print(f"(Không thể tự động hiển thị biểu đồ: {e})")

    # Biểu đồ 2: Tốc độ
    print("Đang tạo biểu đồ so sánh Tốc độ...")
    df_speed = df.sort_values(by="Avg. Time (ms/ảnh)", ascending=True) 
    plt.figure(figsize=(10, 6))
    plt.bar(df_speed["Model"], df_speed["Avg. Time (ms/ảnh)"], color="lightgreen")
    plt.title("So sánh tốc độ dự đoán (càng thấp càng tốt)")
    plt.ylabel("Thời gian trung bình (ms/ảnh)")
    plt.xlabel("Model")
    plt.grid(axis="y", linestyle='--', alpha=0.7)
    plt.tight_layout()
    png_path_speed = os.path.join(MODEL_SAVE_DIR, "model_comparison_speed.png")
    plt.savefig(png_path_speed)
    print(f"Đã lưu biểu đồ Tốc độ vào: {png_path_speed}")
    try:
        plt.show() # CỐ GẮNG HIỂN THỊ BIỂU ĐỒ
    except Exception as e:
        print(f"(Không thể tự động hiển thị biểu đồ: {e})")
        
    # Biểu đồ 3: Tỷ lệ (Biểu đồ tròn)
    print("Đang tạo biểu đồ Tỷ lệ thời gian dự đoán...")
    plt.figure(figsize=(8, 8))
    plt.pie(df["Avg. Time (ms/ảnh)"], labels=df["Model"], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title("Tỷ lệ thời gian dự đoán của các mô hình")
    plt.axis('equal') # Đảm bảo biểu đồ tròn
    png_path_pie = os.path.join(MODEL_SAVE_DIR, "model_comparison_speed_pie.png")
    plt.savefig(png_path_pie)
    print(f"Đã lưu biểu đồ Tỷ lệ (Pie chart) vào: {png_path_pie}")
    try:
        plt.show() # CỐ GẮNG HIỂN THỊ BIỂU ĐỒ
    except Exception as e:
        print(f"(Không thể tự động hiển thị biểu đồ: {e})")
    
    print("\n--- HOÀN TẤT ---")

# =========================================================
# 🏁 MAIN
# =========================================================
if __name__ == "__main__":
    evaluate_all_models()