from flask import Flask, request, jsonify, render_template, redirect
import cv2
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
import tempfile
import base64
import shutil
from image_quality_checker import check_image_quality, get_image_quality_score


# Use a cross-platform temp upload folder
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jfif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)

# Set the upload folder as a configuration variable
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained models (prefer local models in project root)
resnet_model = None
mobilenet_model = None
model_source_path = None
model_load_error = None

# Model configuration
MODEL_TYPE = os.environ.get('MODEL_TYPE', 'resnet')  # 'resnet' or 'mobilenet'
IMG_SIZE_RESNET = 256
IMG_SIZE_MOBILENET = 224

def load_model(model_type='resnet'):
    """
    Load model based on type
    """
    global resnet_model, mobilenet_model, model_source_path, model_load_error
    
    if model_type == 'resnet':
        model_paths = [
            os.environ.get('RESNET_MODEL_PATH'),
            os.path.join(os.path.dirname(__file__), 'resnet152_model.h5'),
            "/home/umang.rathi/Documents/Major Project/resnet152_model.h5"
        ]
        model_name = "ResNet152V2"
    else:  # mobilenet
        model_paths = [
            os.environ.get('MOBILENET_MODEL_PATH'),
            os.path.join(os.path.dirname(__file__), 'mobilenetv2_model.h5'),
            os.path.join(os.path.dirname(__file__), 'mobilenetv2_finetuned_model.h5')
        ]
        model_name = "MobileNetV2"
    
    candidate_paths = [p for p in model_paths if p]
    chosen_path = None
    
    for path in candidate_paths:
        if os.path.exists(path):
            chosen_path = path
            break
    
    if chosen_path is None:
        model_source_path = model_paths[1]  # local path
        print(f'Warning: {model_name} model file not found at any known location')
        return None
    
    model_source_path = chosen_path
    try:
        # First try to load directly
        model = tf.keras.models.load_model(chosen_path)
        print(f'{model_name} model loaded successfully from: {chosen_path}')
        return model
    except Exception as direct_err:
        # On Windows/OneDrive with Unicode paths, try copying to ASCII temp path
        try:
            tmp_dir = os.path.join(tempfile.gettempdir(), f'rice_{model_type}_model')
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, f'{model_type}_model.h5')
            shutil.copy2(chosen_path, tmp_path)
            model = tf.keras.models.load_model(tmp_path)
            model_source_path = tmp_path
            print(f'{model_name} model loaded successfully from temp path: {tmp_path}')
            return model
        except Exception as copy_err:
            error_msg = f"Direct load error: {direct_err}; temp copy load error: {copy_err}"
            model_load_error = error_msg
            print(f'Error loading {model_name} model: {error_msg}')
            return None

# Load models
try:
    resnet_model = load_model('resnet')
    mobilenet_model = load_model('mobilenet')
except Exception as e:
    model_load_error = str(e)
    print('Error loading models:', e)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        # Show a friendly message on the page instead of JSON
        return render_template('index.html', error='Không tìm thấy phần tải tệp (file).'), 400
    file = request.files['file']
    if file.filename == '':
        # Show a friendly message on the page instead of JSON
        return render_template('index.html', error='Bạn chưa chọn tệp hình ảnh.'), 400
    
    # Get model type from request (default to current MODEL_TYPE)
    model_type = request.form.get('model_type', MODEL_TYPE)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Kiểm tra chất lượng ảnh trước khi xử lý (nếu không bị tắt)
        disable_quality_check = request.form.get('disable_quality_check', 'false').lower() == 'true'
        
        if not disable_quality_check:
            print("Đang kiểm tra chất lượng ảnh...")
            quality_results = check_image_quality(filepath)
            quality_score = get_image_quality_score(filepath)
        else:
            print("Đã tắt kiểm tra chất lượng ảnh")
            quality_results = {'overall_valid': True, 'errors': [], 'warnings': [], 'recommendations': []}
            quality_score = {'score': 100, 'grade': 'A+'}
        
        # Kiểm tra xem người dùng có muốn bỏ qua kiểm tra chất lượng không
        skip_quality_check = request.form.get('skip_quality_check', 'false').lower() == 'true'
        disable_quality_check = request.form.get('disable_quality_check', 'false').lower() == 'true'
        
        # Nếu ảnh không đạt chất lượng và người dùng không bỏ qua, hiển thị cảnh báo
        if not quality_results['overall_valid'] and not skip_quality_check and not disable_quality_check:
            error_message = "Ảnh không đạt yêu cầu chất lượng:\n"
            for error in quality_results['errors']:
                error_message += f"• {error}\n"
            error_message += "\nKhuyến nghị:\n"
            for rec in quality_results['recommendations']:
                error_message += f"• {rec}\n"
            error_message += "\n💡 Bạn có thể thử phân tích ảnh này bằng cách chọn 'Bỏ qua kiểm tra chất lượng' (kết quả có thể không chính xác)."
            return render_template('index.html', error=error_message, quality_info=quality_score, show_skip_option=True, selected_model=model_type)
        
        # Nếu có cảnh báo, vẫn cho phép xử lý nhưng hiển thị thông báo
        quality_warnings = []
        if quality_results['warnings']:
            quality_warnings = quality_results['warnings']

        try:
            img = cv2.imread(filepath)
            if img is not None:
                # Resize based on model type
                if model_type == 'mobilenet':
                    img = cv2.resize(img, (IMG_SIZE_MOBILENET, IMG_SIZE_MOBILENET))
                else:  # resnet
                    img = cv2.resize(img, (IMG_SIZE_RESNET, IMG_SIZE_RESNET))
                
                img = img.astype('float32') / 255
                img = np.expand_dims(img, axis=0)
                print(f"Image shape for {model_type}: {img.shape}")
            else:
                return render_template('index.html', error='Không thể đọc ảnh. Vui lòng thử ảnh khác.')
        except Exception as e:
            return render_template('index.html', error=f'Lỗi xử lý ảnh: {str(e)}')

    # Select model based on type
    model = None
    model_name = ""
    if model_type == 'mobilenet':
        model = mobilenet_model
        model_name = "MobileNetV2"
    else:  # resnet
        model = resnet_model
        model_name = "ResNet152V2"

    # Make a prediction on the input image
    if model is None:
        detail = f'{model_name} model chưa được tải trên máy chủ.'
        if model_source_path:
            detail += f" Đường dẫn đang dùng: {model_source_path}."
        if model_load_error:
            detail += f" Lỗi khi tải: {model_load_error}"
        detail += " Xem /health để chẩn đoán."
        return render_template('index.html', error=detail), 500
    
    prediction = model.predict(img)

    # Get the predicted class
    predicted_class = np.argmax(prediction, axis=1)

    # Load the class names
    class_names = ['Bacterial Leaf Blast', 'Brown Spot', 'Healthy', 'Leaf Blast', 'Leaf Scald', 'Narrow Brown Spot']

    # Print the predicted class
    response = {
        'class': class_names[predicted_class[0]],
        'model_used': model_name,
        'confidence': float(np.max(prediction))
    }

    # Encode uploaded image as data URL to show back on page
    try:
        with open(filepath, 'rb') as f:
            image_bytes = f.read()
        ext = os.path.splitext(filepath)[1].lower()
        mime = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        data_url = f"data:{mime};base64,{image_base64}"
    except Exception:
        data_url = None

    # Thêm thông tin chất lượng ảnh vào response
    response['quality_info'] = quality_score
    response['quality_warnings'] = quality_warnings if 'quality_warnings' in locals() else []

    return render_template('index.html', result=response, uploaded_image=data_url, selected_model=model_type)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    resnet_local_path = os.path.join(os.path.dirname(__file__), 'resnet152_model.h5')
    mobilenet_local_path = os.path.join(os.path.dirname(__file__), 'mobilenetv2_model.h5')
    mobilenet_finetuned_path = os.path.join(os.path.dirname(__file__), 'mobilenetv2_finetuned_model.h5')
    fallback_path = "/home/umang.rathi/Documents/Major Project/resnet152_model.h5"
    
    return jsonify({
        'resnet_model_loaded': resnet_model is not None,
        'mobilenet_model_loaded': mobilenet_model is not None,
        'current_model_type': MODEL_TYPE,
        'tf_version': tf.__version__,
        'keras_version': tf.keras.__version__,
        'cwd': os.getcwd(),
        'resnet_file_exists_local': os.path.exists(resnet_local_path),
        'mobilenet_file_exists_local': os.path.exists(mobilenet_local_path),
        'mobilenet_finetuned_exists_local': os.path.exists(mobilenet_finetuned_path),
        'file_exists_fallback': os.path.exists(fallback_path),
        'model_source_path': model_source_path,
        'model_load_error': model_load_error,
    })

if __name__ == '__main__':
    # Run in non-debug mode for stable local testing
    app.run(host='127.0.0.1', port=5000, debug=False)