from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename
import tempfile
import base64
import shutil

# ✅ Import kiểm tra chất lượng ảnh
from image_quality_checker import check_image_quality, get_image_quality_score

# ====================================================
# ⚙️ Cấu hình
# ====================================================
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jfif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ====================================================
# 🧠 Biến model toàn cục
# ====================================================
resnet_model = None
resnet_best_model = None  
mobilenet_model = None
model_source_path = None
model_load_error = None

# ====================================================
# ✅ Hàm load model an toàn
# ====================================================
def load_custom_model(model_type='resnet'):
    global resnet_model, mobilenet_model, resnet_best_model, model_source_path, model_load_error
    
    model_paths = []
    model_name = ""

    if model_type == 'resnet':
        model_paths = [
            os.environ.get('RESNET_MODEL_PATH'),
            os.path.join(os.path.dirname(__file__), 'resnet152_model.h5'),
        ]
        model_name = "ResNet152V2"
    elif model_type == 'mobilenet':
        model_paths = [
            os.environ.get('MOBILENET_MODEL_PATH'),
            os.path.join(os.path.dirname(__file__), 'mobilenetv2_model.h5'),
            os.path.join(os.path.dirname(__file__), 'mobilenetv2_finetuned_model.h5')
        ]
        model_name = "MobileNetV2"
    elif model_type == 'resnet_best':
        model_paths = [
            os.environ.get('RESNET_BEST_MODEL_PATH'),
            os.path.join(os.path.dirname(__file__), 'resnet152v2_rice_disease_best.h5'),
        ]
        model_name = "ResNet152V2 (Best)"
    else:
        print(f"⚠️ Unknown model type: {model_type}")
        return None

    # Tìm file hợp lệ
    for path in model_paths:
        if path and os.path.exists(path):
            model_source_path = path
            try:
                model = load_model(path, compile=False)
                print(f"✅ {model_name} loaded from: {path}")
                return model
            except Exception as e:
                print(f"❌ Error loading {model_name} from {path}: {e}")
                continue

    model_load_error = f"{model_name} not found in any known location"
    print(f"⚠️ {model_load_error}")
    return None

# ====================================================
# 🔁 Load models
# ====================================================
try:
    resnet_model = load_custom_model('resnet')
    mobilenet_model = load_custom_model('mobilenet')
    resnet_best_model = load_custom_model('resnet_best')
except Exception as e:
    model_load_error = str(e)
    print('❌ Error loading models:', e)

# ====================================================
# ⚙️ Cấu hình chung
# ====================================================
MODEL_TYPE = os.environ.get('MODEL_TYPE', 'mobilenet')  # mặc định dùng MobileNet
CLASS_NAMES = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy', 'Leaf Blast']

# ====================================================
# 🔮 Route chính
# ====================================================
@app.route('/predict', methods=['POST'])
def predict():
    global resnet_model, mobilenet_model, resnet_best_model  

    if 'file' not in request.files:
        return render_template('index.html', error='Không tìm thấy phần tải tệp.'), 400

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='Bạn chưa chọn hình ảnh.'), 400

    model_type = request.form.get('model_type', MODEL_TYPE)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # ✅ Kiểm tra chất lượng ảnh
        disable_quality_check = request.form.get('disable_quality_check', 'false').lower() == 'true'
        skip_quality_check = request.form.get('skip_quality_check', 'false').lower() == 'true'

        if not disable_quality_check:
            print("🔍 Đang kiểm tra chất lượng ảnh...")
            quality_results = check_image_quality(filepath)
            quality_score = get_image_quality_score(filepath)
        else:
            print("🛑 Đã tắt kiểm tra chất lượng ảnh.")
            quality_results = {'overall_valid': True, 'errors': [], 'warnings': [], 'recommendations': []}
            quality_score = {'score': 100, 'grade': 'A+'}

        if not quality_results['overall_valid'] and not skip_quality_check:
            error_message = "Ảnh không đạt yêu cầu chất lượng:\n"
            for err in quality_results['errors']:
                error_message += f"• {err}\n"
            error_message += "\nKhuyến nghị:\n"
            for rec in quality_results['recommendations']:
                error_message += f"• {rec}\n"
            return render_template('index.html', error=error_message, quality_info=quality_score, show_skip_option=True)

        # ✅ Tiền xử lý ảnh
        try:
            img = cv2.imread(filepath)
            if img is None:
                return render_template('index.html', error='Không thể đọc ảnh.')
            size = (224, 224) if model_type == 'mobilenet' else (256, 256)
            img = cv2.resize(img, size)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
        except Exception as e:
            return render_template('index.html', error=f'Lỗi xử lý ảnh: {str(e)}')

        # ✅ Chọn model
        model_map = {
            'mobilenet': mobilenet_model,
            'resnet': resnet_model,
            'resnet_best': resnet_best_model
        }
        model = model_map.get(model_type, mobilenet_model)
        model_name = model_type.capitalize()

        if model is None:
            return render_template('index.html', error=f'Model {model_name} chưa được tải.'), 500

        # ✅ Dự đoán
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))

        # ✅ Encode ảnh để hiển thị
        try:
            with open(filepath, 'rb') as f:
                img_bytes = f.read()
            mime = 'image/jpeg' if filename.lower().endswith(('jpg', 'jpeg')) else 'image/png'
            image_base64 = base64.b64encode(img_bytes).decode('utf-8')
            data_url = f"data:{mime};base64,{image_base64}"
        except Exception:
            data_url = None

        try:
            os.remove(filepath)
        except Exception:
            pass

        return render_template('index.html', result={
            'class': CLASS_NAMES[predicted_class],
            'confidence': round(confidence * 100, 2),
            'model_used': model_name,
            'quality_info': quality_score,
            'quality_warnings': quality_results.get('warnings', [])
        }, uploaded_image=data_url)

    return render_template('index.html', error='Tệp không hợp lệ.')

# ====================================================
# 🏠 Trang chủ
# ====================================================
@app.route('/')
def home():
    return render_template('index.html')

# ====================================================
# ❤️ Route health check
# ====================================================
@app.route('/health')
def health():
    return jsonify({
        'resnet_model_loaded': resnet_model is not None,
        'mobilenet_model_loaded': mobilenet_model is not None,
        'resnet_best_model_loaded': resnet_best_model is not None,
        'tensorflow_version': tf.__version__,
        'keras_version': tf.keras.__version__,
        'model_source_path': model_source_path,
        'model_load_error': model_load_error
    })

# ====================================================
# ▶️ Chạy app - CPU ONLY
# ====================================================
if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')  # 🚫 Không dùng GPU
    print("🚀 Flask app running on CPU only...")
    app.run(host='127.0.0.1', port=5000, debug=False)
