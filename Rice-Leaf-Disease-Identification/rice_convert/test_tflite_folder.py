import os
import numpy as np
from PIL import Image
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ------------ CONFIG ------------
TFLITE_MODEL = "rice_leaf_disease.tflite"
TEST_DIR = "test_images"
IMG_SIZE = (256, 256)

# --------------------------------
# ĐẢM BẢO ĐÚNG THỨ TỰ LỚP MÀ MÔ HÌNH ĐÃ HUẤN LUYỆN
class_names = [
    'Bacterial leaf bright',
    'Brown Spot',
    'Healthy',
    'Leaf Blast'
]
# --------------------------------


# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded:", TFLITE_MODEL)
print("Classes used for testing:", class_names)


# Chuẩn bị 2 danh sách để lưu kết quả
y_true = [] # Danh sách nhãn Thật (Ground Truth)
y_pred = [] # Danh sách nhãn Dự đoán (Prediction)


print("\n----- START TESTING -----\n")

for class_name in class_names:
    class_folder = os.path.join(TEST_DIR, class_name)
    if not os.path.isdir(class_folder):
        print(f"LỖI: Không tìm thấy thư mục cho lớp: {class_name}")
        continue

    print(f"🔍 Testing class: {class_name}")
    
    image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(("jpg", "jpeg", "png"))]
    
    if not image_files:
        print(f" - Không tìm thấy ảnh nào trong thư mục {class_name}")
        continue

    for fname in image_files:
        img_path = os.path.join(class_folder, fname)

        # Load ảnh
        try:
            img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
            input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
        except Exception as e:
            print(f" - Lỗi khi đọc ảnh {fname}: {e}")
            continue

        # Nếu INT8 thì quantize
        if input_details[0]['dtype'] in [np.uint8, np.int8]:
            scale, zero_point = input_details[0]['quantization']
            input_data = input_data / scale + zero_point
            input_data = input_data.astype(input_details[0]['dtype'])

        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        pred_idx = np.argmax(output)
        pred_class = class_names[pred_idx]
        
        # Thêm kết quả vào 2 danh sách
        y_true.append(class_name)
        y_pred.append(pred_class)

        # In kết quả từng ảnh (có thể comment dòng này nếu bạn có quá nhiều ảnh)
        print(f" - {fname}: predicted = {pred_class}, true = {class_name}")


if not y_true: # Kiểm tra xem có ảnh nào được xử lý không
    print("\n---------------------------------------")
    print("LỖI: Không có ảnh nào được xử lý. Hãy kiểm tra lại đường dẫn và tên thư mục.")
    print("---------------------------------------")
else:
    # -----------------------------------------------------------------
    # PHẦN NÂNG CẤP: IN BÁO CÁO VÀ VẼ BIỂU ĐỒ
    # -----------------------------------------------------------------
    
    print("\n========================================")
    print("     BÁO CÁO PHÂN LOẠI (Classification Report)    ")
    print("========================================")
    
    # In báo cáo (precision, recall, f1-score)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    print("\n========================================")
    print("     MA TRẬN NHẦM LẪN (Confusion Matrix)    ")
    print("========================================")

    # 1. Tính toán ma trận
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    print(cm)
    print("\nĐang vẽ biểu đồ ma trận nhầm lẫn...")

    # 2. Vẽ biểu đồ heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label (Nhãn thật)')
    plt.xlabel('Predicted Label (Nhãn dự đoán)')
    plt.tight_layout()
    
    # Lưu biểu đồ ra file
    plt.savefig("ket_qua_moi.png")
    print(f"✅ Đã lưu biểu đồ vào file 'ket_qua_moi.png'")
    
    # Hiển thị biểu đồ (nếu bạn muốn nó tự động bật lên)
    # plt.show()
    
    # Tính toán và in độ chính xác tổng thể (lấy từ báo cáo cho dễ)
    accuracy = np.mean(np.array(y_true) == np.array(y_pred)) * 100
    print("\n---------------------------------------")
    print(f"🎉 FINAL ACCURACY: {accuracy:.2f}% ({np.sum(np.array(y_true) == np.array(y_pred))}/{len(y_true)})")
    print("---------------------------------------")