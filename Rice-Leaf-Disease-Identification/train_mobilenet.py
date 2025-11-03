import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator# ✅ dùng keras gốc để tránh cảnh báo
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 🔧 CẤU HÌNH CƠ BẢN
# =========================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Ngắt TF32 để tính toán chính xác hơn trên GPU NVIDIA
tf.config.experimental.enable_tensor_float_32_execution(False)

# Đường dẫn dataset
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/validation'
TEST_DIR = 'dataset/test'

# Danh sách lớp
CLASS_NAMES = ['Bacterial leaf bright', 'Brown Spot', 'Healthy', 'Leaf Blast']
NUM_CLASSES = len(CLASS_NAMES)

# =========================================================
# 🧠 TẠO MÔ HÌNH MOBILENETV2
# =========================================================
def create_mobilenet_model():
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False  # chỉ train phần đầu lúc đầu

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


# =========================================================
# 📦 DATA GENERATOR
# =========================================================
def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator


# =========================================================
# 🚀 TRAIN MODEL
# =========================================================
def train_model():
    print("🔧 Đang tạo mô hình MobileNetV2...")
    model = create_mobilenet_model()

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"✅ Mô hình đã sẵn sàng! Tổng số tham số: {model.count_params():,}")

    train_gen, val_gen, test_gen = create_data_generators()

    os.makedirs("models", exist_ok=True)
    callbacks = [
        ModelCheckpoint('models/mobilenetv2_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]

    print("🏋️‍♂️ Bắt đầu huấn luyện...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    return model, history, test_gen


# =========================================================
# 📊 ĐÁNH GIÁ MÔ HÌNH
# =========================================================
def evaluate_model(model, test_generator):
    print("📈 Đang đánh giá mô hình...")
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    print("\n=== Classification Report ===")
    print(classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES))

    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - MobileNetV2')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('models/mobilenetv2_confusion_matrix.png')
    plt.show()


# =========================================================
# 📉 VẼ BIỂU ĐỒ HUẤN LUYỆN
# =========================================================
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('models/mobilenetv2_training_history.png')
    plt.show()


# =========================================================
# 🔧 FINE-TUNE MÔ HÌNH
# =========================================================
def fine_tune_model(model):
    print("🎯 Bắt đầu fine-tuning MobileNetV2...")
    base_model = model.layers[0]
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"Số lớp được train: {sum([1 for layer in base_model.layers if layer.trainable])}")
    return model


# =========================================================
# 🏁 HÀM MAIN
# =========================================================
def main():
    print("=== TRAINING MOBILE NET V2 FOR RICE LEAF DISEASE DETECTION ===")

    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Không tìm thấy dataset tại: {TRAIN_DIR}")
        print("""
Vui lòng tổ chức dataset như sau:
dataset/
├── train/
│   ├── Bacterial leaf bright/
│   ├── Brown Spot/
│   ├── Healthy/
│   └── Leaf Blast/
├── validation/
└── test/
""")
        return

    try:
        model, history, test_gen = train_model()
        plot_training_history(history)
        evaluate_model(model, test_gen)

        ans = input("\nBạn có muốn fine-tune mô hình không? (y/n): ").lower()
        if ans == 'y':
            model = fine_tune_model(model)
            train_gen, val_gen, _ = create_data_generators()
            model.fit(
                train_gen,
                epochs=10,
                validation_data=val_gen,
                callbacks=[ModelCheckpoint('models/mobilenetv2_finetuned_model.h5',
                                           monitor='val_accuracy', save_best_only=True)],
                verbose=1
            )
            evaluate_model(model, test_gen)

        print("\n✅ Huấn luyện hoàn tất!")
        print("📁 Mô hình đã lưu tại: models/mobilenetv2_model.h5")

    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình huấn luyện: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
