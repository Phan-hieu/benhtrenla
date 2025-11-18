#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import json

# =========================================================
# 🔧 CẤU HÌNH CƠ BẢN
# =========================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0005

# Tắt TF32 nếu có (ổn định trên một số GPU mới)
try:
    tf.config.experimental.enable_tensor_float_32_execution(False)
except Exception:
    pass

TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/validation'
TEST_DIR = 'dataset/test'

# CLASS_NAMES sẽ được cập nhật tự động sau khi tạo train_generator
CLASS_NAMES = None
NUM_CLASSES = 4


# =========================================================
# 🧠 TẠO MÔ HÌNH DENSENET121
# =========================================================
def create_densenet_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES):
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model


# =========================================================
# 📦 DATA GENERATOR (CẬP NHẬT CLASS_NAMES TỰ ĐỘNG)
# =========================================================
def create_data_generators(img_size=IMG_SIZE, batch_size=BATCH_SIZE):
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
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Cập nhật CLASS_NAMES dựa trên class_indices (thứ tự load từ thư mục)
    global CLASS_NAMES, NUM_CLASSES
    CLASS_NAMES = list(train_generator.class_indices.keys())
    NUM_CLASSES = len(CLASS_NAMES)

    print('\n[INFO] Class mapping (index: class):')
    for name, idx in train_generator.class_indices.items():
        print(f'  {idx}: {name}')

    return train_generator, val_generator, test_generator


# =========================================================
# 🚀 TRAIN MODEL
# =========================================================
def train_model():
    print('🔧 Đang tạo mô hình DenseNet121...')
    model = create_densenet_model(num_classes=NUM_CLASSES)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f'✅ Mô hình đã sẵn sàng! Tổng số tham số: {model.count_params():,}')

    train_gen, val_gen, test_gen = create_data_generators()

    # Ensure models folder exists and save class names so serving app can load the same mapping
    os.makedirs('models', exist_ok=True)
    try:
        with open(os.path.join('models', 'class_names.json'), 'w', encoding='utf-8') as f:
            json.dump(CLASS_NAMES, f, ensure_ascii=False)
        print(f"[INFO] Saved class names to models/class_names.json: {CLASS_NAMES}")
    except Exception as e:
        print(f"[WARN] Could not save class names: {e}")
    # Compute class weights to handle any residual imbalance
    labels = train_gen.classes
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(cw))
    print(f"[INFO] Computed class_weights: {class_weights}")

    callbacks = [
        ModelCheckpoint('models/densenet121_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ]

    print('🏋️‍♂️ Bắt đầu huấn luyện DenseNet121...')
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    return model, history, test_gen


# =========================================================
# 📊 ĐÁNH GIÁ MÔ HÌNH
# =========================================================
def evaluate_model(model, test_generator):
    print('📈 Đang đánh giá mô hình...')
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    print('\n=== Classification Report ===')
    print(classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES))

    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - DenseNet121')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('models/densenet121_confusion_matrix.png')
    plt.show()


# =========================================================
# 📉 VẼ BIỂU ĐỒ HUẤN LUYỆN
# =========================================================
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history.get('accuracy', []), label='Training Accuracy')
    ax1.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax2.plot(history.history.get('loss', []), label='Training Loss')
    ax2.plot(history.history.get('val_loss', []), label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('models/densenet121_training_history.png')
    plt.show()


# =========================================================
# 🔧 FINE-TUNE MÔ HÌNH
# =========================================================
def fine_tune_model(model, unfreeze_layers=50):
    print('🎯 Bắt đầu fine-tuning DenseNet121...')
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        base_model = model.layers[0]

    base_model.trainable = True
    fine_tune_at = max(0, len(base_model.layers) - unfreeze_layers)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f'Số lớp được train: {sum([1 for layer in base_model.layers if layer.trainable])}')
    return model


# =========================================================
# 🏁 HÀM MAIN
# =========================================================
def main():
    print('=== TRAINING DENSENET121 FOR RICE LEAF DISEASE DETECTION ===')

    if not os.path.exists(TRAIN_DIR):
        print(f'❌ Không tìm thấy dataset tại: {TRAIN_DIR}')
        print('''
Vui lòng tổ chức dataset như sau:
dataset/
├── train/
│   ├── Bacterial_leaf_bright/
│   ├── Brown Spot/
│   ├── Healthy/
│   └── Leaf Blast/
├── validation/
└── test/
''')
        return

    try:
        model, history, test_gen = train_model()
        plot_training_history(history)
        evaluate_model(model, test_gen)

        ans = input('\nBạn có muốn fine-tune mô hình không? (y/n): ').lower()
        if ans == 'y':
            model = fine_tune_model(model, unfreeze_layers=80)
            train_gen, val_gen, _ = create_data_generators()
            model.fit(
                train_gen,
                epochs=10,
                validation_data=val_gen,
                callbacks=[ModelCheckpoint('models/densenet121_finetuned_model.h5', monitor='val_accuracy', save_best_only=True)],
                verbose=1
            )
            evaluate_model(model, test_gen)

        print('\n✅ Huấn luyện hoàn tất!')
        print('📁 Mô hình đã lưu tại: models/densenet121_model.h5')

    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình huấn luyện: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
