import tensorflow as tf
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình
IMG_SIZE = 256  # ResNet152V2 input size
BATCH_SIZE = 16  # Nhỏ hơn vì ResNet152V2 nặng hơn
EPOCHS = 50
LEARNING_RATE = 0.001

# Đường dẫn dataset
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/validation'
TEST_DIR = 'dataset/test'

# Tên các lớp bệnh
CLASS_NAMES = ['Bacterial Leaf Blast', 'Brown Spot', 'Healthy', 'Leaf Blast', 'Leaf Scald', 'Narrow Brown Spot']
NUM_CLASSES = len(CLASS_NAMES)

def create_attention_layer(input_tensor):
    """
    Tạo attention layer để tập trung vào các phần quan trọng của ảnh
    """
    # Global Average Pooling
    gap = GlobalAveragePooling2D()(input_tensor)
    
    # Dense layer để tạo attention weights
    attention_weights = Dense(1, activation='sigmoid')(gap)
    
    # Reshape để match với input tensor
    attention_weights = tf.keras.layers.Reshape((1, 1, -1))(attention_weights)
    
    # Apply attention weights
    attended_features = tf.keras.layers.Multiply()([input_tensor, attention_weights])
    
    return attended_features

def create_resnet_model():
    """
    Tạo mô hình ResNet152V2 với attention layer
    """
    # Load pre-trained ResNet152V2
    base_model = ResNet152V2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Thêm custom layers với attention
    x = base_model.output
    
    # Thêm attention layer
    x = create_attention_layer(x)
    
    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Tạo model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def create_data_generators():
    """
    Tạo data generators với data augmentation
    """
    # Data augmentation cho training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Chỉ rescale cho validation và test
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    # Validation generator
    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Test generator
    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def train_model():
    """
    Train mô hình ResNet152V2
    """
    print("Đang tạo mô hình ResNet152V2...")
    model = create_resnet_model()
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_3_accuracy']
    )
    
    print("Mô hình được tạo thành công!")
    print(f"Số lượng tham số: {model.count_params():,}")
    
    # Tạo data generators
    print("Đang tạo data generators...")
    train_gen, val_gen, test_gen = create_data_generators()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'resnet152_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("Bắt đầu training...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history, test_gen

def evaluate_model(model, test_generator):
    """
    Đánh giá mô hình trên test set
    """
    print("Đang đánh giá mô hình...")
    
    # Predictions
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - ResNet152V2', fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('resnet152_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Top-3 accuracy
    top3_accuracy = tf.keras.metrics.top_k_categorical_accuracy(
        tf.keras.utils.to_categorical(true_classes, NUM_CLASSES), 
        predictions, 
        k=3
    )
    print(f"\nTop-3 Accuracy: {np.mean(top3_accuracy):.4f}")
    
    return predictions, predicted_classes, true_classes

def plot_training_history(history):
    """
    Vẽ biểu đồ training history
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Top-3 Accuracy
    ax3.plot(history.history['top_3_accuracy'], label='Training Top-3 Accuracy', linewidth=2)
    ax3.plot(history.history['val_top_3_accuracy'], label='Validation Top-3 Accuracy', linewidth=2)
    ax3.set_title('Top-3 Accuracy', fontsize=14)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Top-3 Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Learning Rate (if available)
    if 'lr' in history.history:
        ax4.plot(history.history['lr'], label='Learning Rate', linewidth=2)
        ax4.set_title('Learning Rate Schedule', fontsize=14)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Learning Rate\nNot Available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Learning Rate Schedule', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('resnet152_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def fine_tune_model(model):
    """
    Fine-tune mô hình bằng cách unfreeze một số layers cuối
    """
    print("Bắt đầu fine-tuning...")
    
    # Unfreeze top layers của base model
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze tất cả layers trừ 50 layers cuối
    fine_tune_at = len(base_model.layers) - 50
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Recompile với learning rate thấp hơn
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_3_accuracy']
    )
    
    print(f"Số layers được train: {sum([1 for layer in base_model.layers if layer.trainable])}")
    
    return model

def analyze_model_performance(model, test_generator):
    """
    Phân tích chi tiết hiệu suất mô hình
    """
    print("\n=== PHÂN TÍCH HIỆU SUẤT MÔ HÌNH ===")
    
    # Test accuracy
    test_loss, test_accuracy, test_top3_accuracy = model.evaluate(test_generator, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Top-3 Accuracy: {test_top3_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Predictions
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = true_classes == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(predicted_classes[class_mask] == i)
            print(f"{class_name}: {class_accuracy:.4f} ({np.sum(class_mask)} samples)")
    
    # Confidence analysis
    max_confidences = np.max(predictions, axis=1)
    print(f"\nConfidence Analysis:")
    print(f"Mean Confidence: {np.mean(max_confidences):.4f}")
    print(f"Std Confidence: {np.std(max_confidences):.4f}")
    print(f"Min Confidence: {np.min(max_confidences):.4f}")
    print(f"Max Confidence: {np.max(max_confidences):.4f}")

def main():
    """
    Hàm chính để train mô hình
    """
    print("=== TRAINING RESNET152V2 MODEL FOR RICE LEAF DISEASE DETECTION ===")
    
    # Kiểm tra dataset
    if not os.path.exists(TRAIN_DIR):
        print(f"Lỗi: Không tìm thấy thư mục dataset tại {TRAIN_DIR}")
        print("Vui lòng tải dataset và tổ chức theo cấu trúc:")
        print("dataset/")
        print("├── train/")
        print("│   ├── Bacterial Leaf Blast/")
        print("│   ├── Brown Spot/")
        print("│   ├── Healthy/")
        print("│   ├── Leaf Blast/")
        print("│   ├── Leaf Scald/")
        print("│   └── Narrow Brown Spot/")
        print("├── validation/")
        print("└── test/")
        return
    
    try:
        # Train model
        model, history, test_gen = train_model()
        
        # Plot training history
        plot_training_history(history)
        
        # Evaluate model
        predictions, pred_classes, true_classes = evaluate_model(model, test_gen)
        
        # Analyze performance
        analyze_model_performance(model, test_gen)
        
        # Fine-tune (optional)
        print("\nBạn có muốn fine-tune mô hình không? (y/n): ", end="")
        if input().lower() == 'y':
            model = fine_tune_model(model)
            
            # Train thêm với fine-tuning
            train_gen, val_gen, _ = create_data_generators()
            history_ft = model.fit(
                train_gen,
                epochs=20,  # Ít epochs hơn cho fine-tuning
                validation_data=val_gen,
                callbacks=[ModelCheckpoint('resnet152_finetuned_model.h5', 
                                         monitor='val_accuracy', 
                                         save_best_only=True)],
                verbose=1
            )
            
            # Evaluate fine-tuned model
            print("\n=== EVALUATION AFTER FINE-TUNING ===")
            predictions_ft, pred_classes_ft, _ = evaluate_model(model, test_gen)
            analyze_model_performance(model, test_gen)
        
        print("\n=== HOÀN THÀNH ===")
        print("Mô hình đã được lưu: resnet152_model.h5")
        print("Biểu đồ training history: resnet152_training_history.png")
        print("Confusion matrix: resnet152_confusion_matrix.png")
        
    except Exception as e:
        print(f"Lỗi trong quá trình training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


