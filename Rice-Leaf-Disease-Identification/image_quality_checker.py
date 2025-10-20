import cv2
import numpy as np
from PIL import Image, ImageStat
import os

class ImageQualityChecker:
    """
    Class để kiểm tra chất lượng ảnh trước khi xử lý
    """
    
    def __init__(self):
        # Cấu hình ngưỡng chất lượng - Điều chỉnh linh hoạt hơn
        self.min_resolution = (150, 150)  # Kích thước tối thiểu - giảm từ 224 xuống 150
        self.max_resolution = (8192, 8192)  # Kích thước tối đa - tăng lên 8192
        self.min_blur_threshold = 10  # Ngưỡng blur rất thấp - chỉ từ chối ảnh cực kỳ mờ
        self.min_brightness = 10  # Độ sáng tối thiểu - giảm xuống 10
        self.max_brightness = 250  # Độ sáng tối đa - tăng lên 250
        self.min_contrast = 5  # Độ tương phản tối thiểu - giảm xuống 5
        
    def check_resolution(self, image_path):
        """
        Kiểm tra độ phân giải của ảnh
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Kiểm tra kích thước tối thiểu
                if width < self.min_resolution[0] or height < self.min_resolution[1]:
                    return {
                        'valid': False,
                        'message': f'Ảnh quá nhỏ ({width}x{height}). Kích thước tối thiểu: {self.min_resolution[0]}x{self.min_resolution[1]}',
                        'resolution': (width, height)
                    }
                
                # Kiểm tra kích thước tối đa
                if width > self.max_resolution[0] or height > self.max_resolution[1]:
                    return {
                        'valid': False,
                        'message': f'Ảnh quá lớn ({width}x{height}). Kích thước tối đa: {self.max_resolution[0]}x{self.max_resolution[1]}',
                        'resolution': (width, height)
                    }
                
                return {
                    'valid': True,
                    'message': f'Độ phân giải phù hợp ({width}x{height})',
                    'resolution': (width, height)
                }
                
        except Exception as e:
            return {
                'valid': False,
                'message': f'Không thể đọc thông tin ảnh: {str(e)}',
                'resolution': None
            }
    
    def check_blur(self, image_path):
        """
        Kiểm tra độ nét của ảnh bằng Laplacian variance
        """
        try:
            # Đọc ảnh bằng OpenCV
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'valid': False,
                    'message': 'Không thể đọc ảnh',
                    'blur_score': 0
                }
            
            # Chuyển sang grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Tính Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Đánh giá độ nét với logic rất linh hoạt
            if laplacian_var < 5:  # Cực kỳ mờ - chỉ từ chối những ảnh thực sự không thể phân tích
                return {
                    'valid': False,
                    'message': f'Ảnh cực kỳ mờ (điểm số: {laplacian_var:.1f}). Không thể phân tích được.',
                    'blur_score': laplacian_var
                }
            elif laplacian_var < 15:  # Mờ nhưng vẫn có thể thử
                return {
                    'valid': True,
                    'message': f'Ảnh hơi mờ (điểm số: {laplacian_var:.1f}) nhưng có thể thử phân tích.',
                    'blur_score': laplacian_var,
                    'warning': True
                }
            elif laplacian_var < 50:  # Chấp nhận được
                return {
                    'valid': True,
                    'message': f'Ảnh có độ nét chấp nhận được (điểm số: {laplacian_var:.1f})',
                    'blur_score': laplacian_var
                }
            else:  # Rõ nét
                return {
                    'valid': True,
                    'message': f'Ảnh rõ nét (điểm số: {laplacian_var:.1f})',
                    'blur_score': laplacian_var
                }
                
        except Exception as e:
            return {
                'valid': False,
                'message': f'Lỗi kiểm tra độ nét: {str(e)}',
                'blur_score': 0
            }
    
    def check_brightness_contrast(self, image_path):
        """
        Kiểm tra độ sáng và tương phản của ảnh
        """
        try:
            with Image.open(image_path) as img:
                # Chuyển sang grayscale để tính toán
                gray_img = img.convert('L')
                
                # Tính độ sáng trung bình
                stat = ImageStat.Stat(gray_img)
                brightness = stat.mean[0]
                
                # Tính độ tương phản (standard deviation)
                contrast = stat.stddev[0]
                
                # Kiểm tra độ sáng
                brightness_issues = []
                if brightness < self.min_brightness:
                    brightness_issues.append(f'Ảnh quá tối (độ sáng: {brightness:.1f})')
                elif brightness > self.max_brightness:
                    brightness_issues.append(f'Ảnh quá sáng (độ sáng: {brightness:.1f})')
                
                # Kiểm tra độ tương phản
                contrast_issues = []
                if contrast < self.min_contrast:
                    contrast_issues.append(f'Ảnh thiếu tương phản (độ tương phản: {contrast:.1f})')
                
                # Tổng hợp kết quả
                all_issues = brightness_issues + contrast_issues
                
                if all_issues:
                    return {
                        'valid': False,
                        'message': '; '.join(all_issues),
                        'brightness': brightness,
                        'contrast': contrast
                    }
                else:
                    return {
                        'valid': True,
                        'message': f'Độ sáng và tương phản phù hợp (sáng: {brightness:.1f}, tương phản: {contrast:.1f})',
                        'brightness': brightness,
                        'contrast': contrast
                    }
                    
        except Exception as e:
            return {
                'valid': False,
                'message': f'Lỗi kiểm tra độ sáng/tương phản: {str(e)}',
                'brightness': 0,
                'contrast': 0
            }
    
    def check_file_size(self, image_path):
        """
        Kiểm tra kích thước file - Linh hoạt hơn
        """
        try:
            file_size = os.path.getsize(image_path)
            file_size_mb = file_size / (1024 * 1024)
            
            # Kích thước file quá nhỏ có thể là ảnh chất lượng thấp
            if file_size_mb < 0.01:  # < 10KB - rất nhỏ
                return {
                    'valid': False,
                    'message': f'File ảnh quá nhỏ ({file_size_mb:.3f}MB). Có thể không phải ảnh hợp lệ.',
                    'file_size_mb': file_size_mb
                }
            elif file_size_mb < 0.05:  # < 50KB - nhỏ nhưng có thể chấp nhận
                return {
                    'valid': True,
                    'message': f'File ảnh nhỏ ({file_size_mb:.3f}MB) nhưng có thể phân tích được.',
                    'file_size_mb': file_size_mb,
                    'warning': True
                }
            elif file_size_mb > 100:  # > 100MB - quá lớn
                return {
                    'valid': False,
                    'message': f'File ảnh quá lớn ({file_size_mb:.2f}MB). Vui lòng chọn ảnh nhỏ hơn.',
                    'file_size_mb': file_size_mb
                }
            else:
                return {
                    'valid': True,
                    'message': f'Kích thước file phù hợp ({file_size_mb:.2f}MB)',
                    'file_size_mb': file_size_mb
                }
                
        except Exception as e:
            return {
                'valid': False,
                'message': f'Lỗi kiểm tra kích thước file: {str(e)}',
                'file_size_mb': 0
            }
    
    def check_image_format(self, image_path):
        """
        Kiểm tra định dạng ảnh
        """
        try:
            with Image.open(image_path) as img:
                format_name = img.format
                
                # Các định dạng được hỗ trợ
                supported_formats = ['JPEG', 'PNG', 'JPG', 'BMP', 'TIFF']
                
                if format_name not in supported_formats:
                    return {
                        'valid': False,
                        'message': f'Định dạng {format_name} không được hỗ trợ. Vui lòng sử dụng JPEG, PNG, BMP hoặc TIFF.',
                        'format': format_name
                    }
                else:
                    return {
                        'valid': True,
                        'message': f'Định dạng {format_name} được hỗ trợ',
                        'format': format_name
                    }
                    
        except Exception as e:
            return {
                'valid': False,
                'message': f'Lỗi kiểm tra định dạng: {str(e)}',
                'format': None
            }
    
    def comprehensive_check(self, image_path):
        """
        Kiểm tra toàn diện chất lượng ảnh
        """
        results = {
            'overall_valid': True,
            'checks': {},
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Kiểm tra định dạng
        format_check = self.check_image_format(image_path)
        results['checks']['format'] = format_check
        if not format_check['valid']:
            results['overall_valid'] = False
            results['errors'].append(format_check['message'])
        
        # Kiểm tra kích thước file
        size_check = self.check_file_size(image_path)
        results['checks']['file_size'] = size_check
        if not size_check['valid']:
            results['overall_valid'] = False
            results['errors'].append(size_check['message'])
        
        # Kiểm tra độ phân giải
        resolution_check = self.check_resolution(image_path)
        results['checks']['resolution'] = resolution_check
        if not resolution_check['valid']:
            results['overall_valid'] = False
            results['errors'].append(resolution_check['message'])
        
        # Kiểm tra độ nét
        blur_check = self.check_blur(image_path)
        results['checks']['blur'] = blur_check
        if not blur_check['valid']:
            results['overall_valid'] = False
            results['errors'].append(blur_check['message'])
        elif blur_check.get('warning'):
            results['warnings'].append(blur_check['message'])
        
        # Kiểm tra độ sáng và tương phản
        brightness_contrast_check = self.check_brightness_contrast(image_path)
        results['checks']['brightness_contrast'] = brightness_contrast_check
        if not brightness_contrast_check['valid']:
            results['overall_valid'] = False
            results['errors'].append(brightness_contrast_check['message'])
        
        # Tạo khuyến nghị
        if results['overall_valid']:
            results['recommendations'].append("Ảnh có chất lượng tốt, có thể tiến hành phân tích.")
        else:
            results['recommendations'].append("Vui lòng chụp lại ảnh với chất lượng tốt hơn.")
        
        # Thêm khuyến nghị cụ thể
        if blur_check.get('blur_score', 0) < self.min_blur_threshold * 2:
            results['recommendations'].append("Đảm bảo ảnh được chụp trong điều kiện ánh sáng tốt và giữ máy ảnh ổn định.")
        
        if resolution_check.get('resolution'):
            width, height = resolution_check['resolution']
            if width < 512 or height < 512:
                results['recommendations'].append("Chụp ảnh với độ phân giải cao hơn để có kết quả tốt hơn.")
        
        return results
    
    def get_quality_score(self, image_path):
        """
        Tính điểm chất lượng tổng thể (0-100)
        """
        try:
            results = self.comprehensive_check(image_path)
            score = 100
            
            # Trừ điểm cho các lỗi
            for check_name, check_result in results['checks'].items():
                if not check_result['valid']:
                    if check_name == 'format':
                        score -= 30
                    elif check_name == 'file_size':
                        score -= 20
                    elif check_name == 'resolution':
                        score -= 25
                    elif check_name == 'blur':
                        # Trừ điểm ít hơn cho blur vì đã linh hoạt hơn
                        score -= 20
                    elif check_name == 'brightness_contrast':
                        score -= 15
            
            # Trừ điểm cho cảnh báo
            score -= len(results['warnings']) * 5
            
            # Đảm bảo điểm không âm
            score = max(0, score)
            
            return {
                'score': score,
                'grade': self._get_quality_grade(score),
                'results': results
            }
            
        except Exception as e:
            return {
                'score': 0,
                'grade': 'F',
                'error': str(e)
            }
    
    def _get_quality_grade(self, score):
        """
        Chuyển điểm số thành grade
        """
        if score >= 90:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 60:
            return 'C'
        elif score >= 50:
            return 'D'
        else:
            return 'F'

# Hàm tiện ích để sử dụng trong Flask app
def check_image_quality(image_path):
    """
    Hàm tiện ích để kiểm tra chất lượng ảnh
    """
    checker = ImageQualityChecker()
    return checker.comprehensive_check(image_path)

def get_image_quality_score(image_path):
    """
    Hàm tiện ích để lấy điểm chất lượng ảnh
    """
    checker = ImageQualityChecker()
    return checker.get_quality_score(image_path)

if __name__ == "__main__":
    # Test với ảnh mẫu
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        checker = ImageQualityChecker()
        
        print("=== KIỂM TRA CHẤT LƯỢNG ẢNH ===")
        print(f"Ảnh: {image_path}")
        print()
        
        # Kiểm tra toàn diện
        results = checker.comprehensive_check(image_path)
        
        print("KẾT QUẢ KIỂM TRA:")
        for check_name, check_result in results['checks'].items():
            status = "✅" if check_result['valid'] else "❌"
            print(f"{status} {check_name.upper()}: {check_result['message']}")
        
        print()
        if results['warnings']:
            print("CẢNH BÁO:")
            for warning in results['warnings']:
                print(f"⚠️ {warning}")
        
        if results['errors']:
            print("LỖI:")
            for error in results['errors']:
                print(f"❌ {error}")
        
        print()
        print("KHUYẾN NGHỊ:")
        for rec in results['recommendations']:
            print(f"💡 {rec}")
        
        # Điểm chất lượng
        quality_score = checker.get_quality_score(image_path)
        print()
        print(f"ĐIỂM CHẤT LƯỢNG: {quality_score['score']}/100 (Grade: {quality_score['grade']})")
        
    else:
        print("Sử dụng: python image_quality_checker.py <đường_dẫn_ảnh>")
