import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# ⚙️ CẤU HÌNH CƠ BẢN
# =========================================================
IMG_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.001

TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/validation'
TEST_DIR = 'dataset/test'

CLASS_NAMES = ['Bacterial leaf bright', 'Brown Spot', 'Healthy', 'Leaf Blast']
NUM_CLASSES = len(CLASS_NAMES)

# =========================================================
# 🧠 TẠO MÔ HÌNH ResNet152V2
# =========================================================
def create_resnet_model():
    base_model = ResNet152V2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model

# =========================================================
# 📦 DATA GENERATOR
# =========================================================
def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

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
    print("🔧 Đang tạo mô hình ResNet152V2...")
    model = create_resnet_model()

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"✅ Mô hình sẵn sàng! Tổng tham số: {model.count_params():,}")

    train_gen, val_gen, test_gen = create_data_generators()

    os.makedirs("models", exist_ok=True)
    callbacks = [
        ModelCheckpoint('models/resnet152v2_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    print("🏋️‍♂️ Huấn luyện trên CPU/GPU...")
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
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    print("\n=== Classification Report ===")
    print(classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES))

    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - ResNet152V2')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('models/resnet152v2_confusion_matrix.png')
    plt.show()

# =========================================================
# 📉 VẼ BIỂU ĐỒ HUẤN LUYỆN
# =========================================================
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend(); plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend(); plt.title('Loss')

    plt.tight_layout()
    plt.savefig('models/resnet152v2_training_history.png')
    plt.show()

# =========================================================
# 🎯 FINE-TUNE CHUẨN
# =========================================================
def fine_tune_model(model):
    print("🎯 Bắt đầu fine-tuning ResNet152V2...")

    base_model = model.get_layer('resnet152v2')
    base_model.trainable = True

    fine_tune_at = len(base_model.layers) - 50
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Số lớp được train: {sum([1 for l in base_model.layers if l.trainable])}")
    return model

# =========================================================
# 🏁 MAIN
# =========================================================
def main():
    print("=== TRAINING ResNet152V2 FOR RICE LEAF DISEASE DETECTION ===")

    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Không tìm thấy dataset tại: {TRAIN_DIR}")
        print("""
Cấu trúc thư mục cần:
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

    model, history, test_gen = train_model()
    plot_training_history(history)
    evaluate_model(model, test_gen)

    ans = input("\nBạn có muốn fine-tune mô hình không? (y/n): ").lower()
    if ans == 'y':
        model = fine_tune_model(model)
        train_gen, val_gen, _ = create_data_generators()

        print("🏋️‍♂️ Fine-tuning bắt đầu...")
        model.fit(
            train_gen,
            epochs=8,
            validation_data=val_gen,
            callbacks=[
                ModelCheckpoint('models/resnet152v2_finetuned.h5', monitor='val_accuracy', save_best_only=True),
                EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)
            ],
            verbose=1
        )
        evaluate_model(model, test_gen)

    print("\n✅ Huấn luyện hoàn tất! Mô hình lưu tại: models/resnet152v2_best.h5")

if __name__ == "__main__":
    main()
