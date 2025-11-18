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
from tensorflow.keras.layers import DepthwiseConv2D
# Giả sử bạn có file này
from image_quality_checker import check_image_quality, get_image_quality_score

# Custom layer definition for model loading
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        super(CustomDepthwiseConv2D, self).__init__(*args, **kwargs)
    
    def get_config(self):
        config = super(CustomDepthwiseConv2D, self).get_config()
        return config


# Wrapper Conv2D để xử lý tên layer có ký tự '/' (một số model cũ lưu tên dạng 'conv1/conv')
class CustomConv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        # Nếu tên layer chứa '/', thay thế bằng '_' để tf.keras chấp nhận
        if 'name' in kwargs and isinstance(kwargs['name'], str):
            kwargs['name'] = kwargs['name'].replace('/', '_')
        super(CustomConv2D, self).__init__(*args, **kwargs)

    def get_config(self):
        config = super(CustomConv2D, self).get_config()
        return config


# Tương tự cho BatchNormalization (một số tên layer cũ có ký tự '/')
class CustomBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, *args, **kwargs):
        if 'name' in kwargs and isinstance(kwargs['name'], str):
            kwargs['name'] = kwargs['name'].replace('/', '_')
        super(CustomBatchNormalization, self).__init__(*args, **kwargs)

    def get_config(self):
        config = super(CustomBatchNormalization, self).get_config()
        return config


# Wrapper cho Activation (tên layer có thể chứa '/')
class CustomActivation(tf.keras.layers.Activation):
    def __init__(self, *args, **kwargs):
        if 'name' in kwargs and isinstance(kwargs['name'], str):
            kwargs['name'] = kwargs['name'].replace('/', '_')
        super(CustomActivation, self).__init__(*args, **kwargs)

    def get_config(self):
        config = super(CustomActivation, self).get_config()
        return config
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

resnet_best_model = None  
mobilenet_model = None
densenet_model = None
densenet201_model = None
# store per-model source paths and any load errors
model_source_paths = {}
model_load_errors = {}

# ====================================================
# ✅ Hàm load model an toàn
# ====================================================
from tensorflow.keras.applications import DenseNet201 as _DenseNet201_base
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense as _Dense, Dropout as _Dropout
from tensorflow.keras.models import Model as _Model
def load_custom_model(model_type='resnet'):
    global mobilenet_model, resnet_best_model, densenet_model, densenet201_model, model_source_paths, model_load_errors
    
    model_paths = []
    model_name = ""

    if model_type == 'mobilenet':
        model_paths = [
            os.environ.get('MOBILENET_MODEL_PATH'),
            os.path.join(os.path.dirname(__file__), 'models', 'mobilenetv2_model.h5')
        ]
        model_name = "MobileNetV2"
    elif model_type == 'resnet_best':
        model_paths = [
            os.environ.get('RESNET_BEST_MODEL_PATH'),
            os.path.join(os.path.dirname(__file__),'models', 'resnet152v2_rice_disease_best.h5'),
        ]
        model_name = "ResNet152V2 (Best)"
    elif model_type == 'densenet':
        model_paths = [
            os.environ.get('DENSENET_MODEL_PATH'),
            os.path.join(os.path.dirname(__file__),'models', 'densenet121_model.h5'),
        ]
        model_name = "DenseNet121"
    elif model_type == 'densenet121': # Có vẻ bạn có 2 key cho densenet121?
        model_paths = [
            os.environ.get('DENSENET121_MODEL_PATH'), # Sửa từ DENSENET201_MODEL_PATH
            os.path.join(os.path.dirname(__file__),'models', 'densenet121_best.h5'),
        ]
        model_name = "DenseNet121"
    elif model_type == 'densenet201':
        model_paths = [
            os.environ.get('DENSENET201_MODEL_PATH'),
            os.path.join(os.path.dirname(__file__), 'models', 'densenet201_best.h5'),
        ]
        model_name = "DenseNet201" # Thêm model_name

    # Tìm file hợp lệ
    for path in model_paths:
        if path and os.path.exists(path):
            model_source_paths[model_type] = path
            try:
                # Custom objects cho việc load model
                custom_objects = {
                    'DepthwiseConv2D': CustomDepthwiseConv2D,
                    'Conv2D': CustomConv2D,
                    'GlorotUniform': tf.keras.initializers.GlorotUniform,
                    'Zeros': tf.keras.initializers.Zeros,
                    'Ones': tf.keras.initializers.Ones,
                    'GlorotNormal': tf.keras.initializers.GlorotNormal,
                    'RandomNormal': tf.keras.initializers.RandomNormal,
                    'BatchNormalization': CustomBatchNormalization,
                    'Activation': CustomActivation,
                }
                model = load_model(path, compile=False, custom_objects=custom_objects)
                model_source_paths[model_type] = path
                model_load_errors.pop(model_type, None)
                print(f"✅ {model_name} loaded from: {path}")
                return model
            except Exception as e:
                # record error and continue to try other candidate paths
                model_load_errors[model_type] = str(e)
                print(f"❌ Error loading {model_name} from {path}: {e}")
                # Nếu là DenseNet201, thử fallback: xây dựng kiến trúc tương đương và load weights
                if model_type == 'densenet201':
                    try:
                        print(f"🔁 Attempting fallback: instantiate DenseNet201 architecture and load weights from {path}...")
                        # create model like training script
                        IMG_SIZE = 224
                        # Cần định nghĩa NUM_CLASSES ở đây, hoặc lấy từ CLASS_NAMES
                        NUM_CLASSES = len(CLASS_NAMES) if 'CLASS_NAMES' in globals() else 4 # Giả sử 4 class
                        base = _DenseNet201_base(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
                        base.trainable = False
                        x = base.output
                        x = GlobalAveragePooling2D()(x)
                        x = _Dense(256, activation='relu')(x)
                        x = _Dropout(0.3)(x)
                        outputs = _Dense(NUM_CLASSES, activation='softmax')(x)
                        fallback_model = _Model(inputs=base.input, outputs=outputs)
                        # Try loading weights (this works if the .h5 contains weights groups)
                        fallback_model.load_weights(path)
                        print(f"✅ DenseNet201 weights loaded into constructed model from: {path}")
                        model_load_errors.pop(model_type, None) # Xóa lỗi nếu fallback thành công
                        return fallback_model
                    except Exception as e2:
                        print(f"❌ Fallback loading DenseNet201 weights failed: {e2}")
                        # continue to next candidate path (if any)
                continue

    if model_type not in model_load_errors:
        model_load_errors[model_type] = f"{model_name} not found in any known location"
    print(f"⚠️ {model_load_errors[model_type]}")
    return None

# ====================================================
# ⚙️ Cấu hình chung (phải load class_names trước)
# ====================================================
DEFAULT_CLASS_NAMES = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy', 'Leaf Blast']
class_names_path = os.path.join(os.path.dirname(__file__), 'models', 'class_names.json')
try:
    if os.path.exists(class_names_path):
        import json
        with open(class_names_path, 'r', encoding='utf-8') as f:
            CLASS_NAMES = json.load(f)
            print(f"[INFO] Loaded CLASS_NAMES from {class_names_path}: {CLASS_NAMES}")
    else:
        CLASS_NAMES = DEFAULT_CLASS_NAMES
        print(f"[WARN] class_names.json not found, using default CLASS_NAMES: {CLASS_NAMES}")
except Exception as e:
    CLASS_NAMES = DEFAULT_CLASS_NAMES
    print(f"[ERROR] Failed to load class names file: {e}. Falling back to defaults: {CLASS_NAMES}")


# ====================================================
# 🔁 Load models
# ====================================================
try:
    mobilenet_model = load_custom_model('mobilenet')
    resnet_best_model = load_custom_model('resnet_best')
    densenet_model = load_custom_model('densenet')
    densenet201_model = load_custom_model('densenet201')
except Exception as e:
    model_load_error = str(e)
    print('❌ Error loading models:', e)

# Determine which models actually loaded and pick a sensible default if the configured default is missing
loaded_models = {
    'mobilenet': mobilenet_model is not None,
    'resnet_best': resnet_best_model is not None,
    'densenet': densenet_model is not None,
    'densenet201': densenet201_model is not None
}
print('[INFO] Loaded models status:', loaded_models)

# Logic chọn MODEL_TYPE (default)
env_default = os.environ.get('MODEL_TYPE', None)
preferred = env_default or 'mobilenet' # Default gốc là mobilenet nếu không có env
MODEL_TYPE = preferred # Bắt đầu với default

# Nếu user không set env và DenseNet201 có sẵn, ưu tiên nó
if env_default is None and loaded_models.get('densenet201'):
    MODEL_TYPE = 'densenet201'
# Nếu default được chọn (từ env hoặc code) không load được
elif preferred and loaded_models.get(preferred) is False:
    # tìm model đầu tiên load được
    first_loaded = next((k for k, v in loaded_models.items() if v), None)
    if first_loaded:
        print(f"[WARN] Preferred MODEL_TYPE='{preferred}' not loaded; switching default to '{first_loaded}'")
        MODEL_TYPE = first_loaded
    else:
        print("[ERROR] No models loaded. The app will still start but predictions will return model-not-loaded errors.")
        MODEL_TYPE = 'mobilenet' # Fallback cuối cùng
# Nếu không, MODEL_TYPE đã được set thành `preferred` (và nó đã load)
print(f"[INFO] Default MODEL_TYPE set to: '{MODEL_TYPE}'")


# ====================================================
# 🔮 Route chính
# ====================================================
@app.route('/predict', methods=['POST'])
def predict():
    global resnet_best_model, mobilenet_model, densenet_model, densenet201_model

    # *** FIX: Lấy model_type từ form ngay lập tức ***
    # Dùng MODEL_TYPE (default toàn cục) chỉ khi 'model_type' không có trong form
    model_type = request.form.get('model_type', MODEL_TYPE)

    if 'file' not in request.files:
        # *** FIX: Trả về model_type mà user đã chọn (hoặc default) ***
        return render_template('index.html', error='Không tìm thấy phần tải tệp.', loaded_models=loaded_models, selected_model=model_type), 400

    file = request.files['file']
    if file.filename == '':
        # *** FIX: Trả về model_type mà user đã chọn (hoặc default) ***
        return render_template('index.html', error='Bạn chưa chọn hình ảnh.', loaded_models=loaded_models, selected_model=model_type), 400

    # Dòng này không cần nữa vì đã chuyển lên đầu
    # model_type = request.form.get('model_type', MODEL_TYPE) 

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # ✅ Kiểm tra chất lượng ảnh
        disable_quality_check = request.form.get('disable_quality_check', 'false').lower() == 'true'
        skip_quality_check = request.form.get('skip_quality_check', 'false').lower() == 'true'

        # ...
        if not disable_quality_check:
            print("🔍 Đang kiểm tra chất lượng ảnh...")
            # GỌI 1 LẦN DUY NHẤT
            quality_data = get_image_quality_score(filepath) 
            
            # Tách dữ liệu ra
            quality_score = {'score': quality_data['score'], 'grade': quality_data['grade']}
            quality_results = quality_data['results']
        else:
            print("🛑 Đã tắt kiểm tra chất lượng ảnh.")
            quality_results = {'overall_valid': True, 'errors': [], 'warnings': [], 'recommendations': []}
            quality_score = {'score': 100, 'grade': 'A+'}

        if not quality_results['overall_valid'] and not skip_quality_check:
            # ...
        # ...
            error_message = "Ảnh không đạt yêu cầu chất lượng:\n"
            for err in quality_results['errors']:
                error_message += f"• {err}\n"
            error_message += "\nKhuyến nghị:\n"
            for rec in quality_results['recommendations']:
                error_message += f"• {rec}\n"
            
            # *** FIX: Trả về model_type mà user đã chọn ***
            return render_template('index.html', error=error_message, quality_info=quality_score, show_skip_option=True, loaded_models=loaded_models, selected_model=model_type)

        # ✅ Tiền xử lý ảnh
        try:
            img = cv2.imread(filepath)
            if img is None:
                # *** FIX: Trả về model_type mà user đã chọn ***
                return render_template('index.html', error='Không thể đọc ảnh.', loaded_models=loaded_models, selected_model=model_type)
            
            # Xử lý kích thước ảnh cho từng loại model
            if model_type in ('mobilenet', 'densenet', 'densenet201'):
                size = (224, 224)
            else: # 'resnet_best' và các model khác
                size = (256, 256) 
                
            img = cv2.resize(img, size)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
        except Exception as e:
            # *** FIX: Trả về model_type mà user đã chọn ***
            return render_template('index.html', error=f'Lỗi xử lý ảnh: {str(e)}', loaded_models=loaded_models, selected_model=model_type)

        # ✅ Chọn model
        model_map = {
            'mobilenet': mobilenet_model,
            'resnet_best': resnet_best_model,
            'densenet': densenet_model,
            'densenet201': densenet201_model
        }
        # Dùng model_map.get(model_type) và kiểm tra None sau
        model = model_map.get(model_type) 
        
        # Tên hiển thị cho từng loại model
        model_display_names = {
            'mobilenet': 'MobileNetV2',
            'resnet_best': 'ResNet152V2 (Best)',
            'densenet': 'DenseNet121',
            'densenet201': 'DenseNet201'
        }
        model_name = model_display_names.get(model_type, model_type.capitalize())

        if model is None:
            # *** FIX: Trả về model_type mà user đã chọn ***
            error_msg = f'Model {model_name} chưa được tải.'
            if model_type in model_load_errors:
                error_msg += f" Lỗi: {model_load_errors[model_type]}"
            return render_template('index.html', error=error_msg, loaded_models=loaded_models, selected_model=model_type), 500

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

        # *** FIX: Trả về model_type mà user đã chọn ***
        return render_template('index.html', result={
            'class': CLASS_NAMES[predicted_class],
            'confidence': round(confidence * 100, 2),
            'model_used': model_name,
            'quality_info': quality_score,
            'quality_warnings': quality_results.get('warnings', [])
        }, uploaded_image=data_url, loaded_models=loaded_models, selected_model=model_type)
    
    # *** FIX: Trả về model_type mà user đã chọn ***
    return render_template('index.html', error='Tệp không hợp lệ.', loaded_models=loaded_models, selected_model=model_type)

# ====================================================
# 🏠 Trang chủ
# ====================================================
@app.route('/')
def home():
    # Trang chủ vẫn dùng MODEL_TYPE (default toàn cục)
    return render_template('index.html', loaded_models=loaded_models, selected_model=MODEL_TYPE)

# ====================================================
# ❤️ Route health check
# ====================================================
@app.route('/health')
def health():
    return jsonify({
        'mobilenet_model_loaded': mobilenet_model is not None,
        'resnet_best_model_loaded': resnet_best_model is not None,
        'densenet_model_loaded': densenet_model is not None,
        'densenet201_model_loaded': densenet201_model is not None,
        'tensorflow_version': tf.__version__,
        'keras_version': tf.keras.__version__,
        'model_source_paths': model_source_paths,
        'model_load_errors': model_load_errors
    })

# ====================================================
# ▶️ Chạy app - CPU ONLY
# ====================================================
if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')  # 🚫 Không dùng GPU
    print("🚀 Flask app running on CPU only...")
    app.run(host='127.0.0.1', port=5000, debug=False)