# convert_tf17.py
import tensorflow as tf
import os

print("TF version:", tf.__version__)
MODEL_H5 = "resnet152v2_rice_disease_best.h5"
OUT_TFLITE = "rice_leaf_disease.tflite"

if not os.path.exists(MODEL_H5):
    raise FileNotFoundError(f"{MODEL_H5} not found in current folder.")

# 1. Load Keras model
# Nếu bạn dùng custom_objects, cung cấp dict custom_objects={...}
model = tf.keras.models.load_model(MODEL_H5)
print("Loaded model:", model)

# 2. (Optional) build model nếu gặp lỗi về concrete function
# Thay input_shape tùy theo model của bạn (thường MobileNetV2 dùng 224x224x3)
try:
    model.build(input_shape=(None, 224, 224, 3))
    print("Model built with input shape (None,224,224,3)")
except Exception as e:
    print("Build skipped or failed:", e)

# 3. Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 4. Save
with open(OUT_TFLITE, "wb") as f:
    f.write(tflite_model)

print(f"✅ Done. Saved: {OUT_TFLITE}")
