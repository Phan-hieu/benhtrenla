// Image Preview Functionality
document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const fileDimensions = document.getElementById('fileDimensions');
    const removeBtn = document.getElementById('removeImage');
    const fileLabel = document.querySelector('.file-label');
    const fileText = document.querySelector('.file-text');

    // Function to format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Function to get image dimensions
    function getImageDimensions(file) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = function () {
                resolve({
                    width: this.naturalWidth,
                    height: this.naturalHeight
                });
            };
            img.onerror = function () {
                resolve({ width: 0, height: 0 });
            };
            img.src = URL.createObjectURL(file);
        });
    }

    // Function to show preview
    function showPreview(file) {
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();

            reader.onload = function (e) {
                previewImg.src = e.target.result;

                // Update file info
                fileName.textContent = file.name;
                fileSize.textContent = `Kích thước: ${formatFileSize(file.size)}`;

                // Get and display dimensions
                getImageDimensions(file).then(dimensions => {
                    if (dimensions.width > 0 && dimensions.height > 0) {
                        fileDimensions.textContent = `Kích thước: ${dimensions.width} x ${dimensions.height} pixels`;
                    } else {
                        fileDimensions.textContent = 'Không thể đọc kích thước ảnh';
                    }
                });

                // Show preview container
                imagePreview.style.display = 'block';
                imagePreview.classList.add('show');

                // Update file label text
                fileText.textContent = 'Thay đổi ảnh';
            };

            reader.readAsDataURL(file);
        }
    }

    // Function to hide preview
    function hidePreview() {
        imagePreview.style.display = 'none';
        imagePreview.classList.remove('show');
        fileInput.value = '';
        previewImg.src = '';
        fileName.textContent = '';
        fileSize.textContent = '';
        fileDimensions.textContent = '';
        fileText.textContent = 'Chọn ảnh lá lúa';
    }

    // Event listener for file input change
    fileInput.addEventListener('change', function (e) {
        const file = e.target.files[0];
        if (file) {
            showPreview(file);
        } else {
            hidePreview();
        }
    });

    // Event listener for remove button
    removeBtn.addEventListener('click', function (e) {
        e.preventDefault();
        hidePreview();
    });

    // Drag and drop functionality
    const fileUpload = document.querySelector('.file-upload');

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        fileUpload.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        fileUpload.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        fileUpload.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    fileUpload.addEventListener('drop', handleDrop, false);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight(e) {
        fileUpload.classList.add('drag-over');
    }

    function unhighlight(e) {
        fileUpload.classList.remove('drag-over');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                fileInput.files = files;
                showPreview(file);
            } else {
                alert('Vui lòng chọn file ảnh (JPEG, PNG, BMP, TIFF)');
            }
        }
    }

    // Add drag over styles
    const style = document.createElement('style');
    style.textContent = `
        .file-upload.drag-over {
            border: 2px dashed #28a745 !important;
            background-color: #f8fff9 !important;
        }
        
        .file-upload.drag-over .file-label {
            background: linear-gradient(45deg, #28a745, #20c997) !important;
        }
    `;
    document.head.appendChild(style);

    // Image quality pre-check (optional)
    function preCheckImageQuality(file) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = function () {
                // Basic quality checks
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = this.naturalWidth;
                canvas.height = this.naturalHeight;
                ctx.drawImage(this, 0, 0);

                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const data = imageData.data;

                // Calculate brightness
                let brightness = 0;
                for (let i = 0; i < data.length; i += 4) {
                    brightness += (data[i] + data[i + 1] + data[i + 2]) / 3;
                }
                brightness /= (data.length / 4);

                // Calculate contrast (simplified)
                let contrast = 0;
                for (let i = 0; i < data.length; i += 4) {
                    const pixelBrightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
                    contrast += Math.abs(pixelBrightness - brightness);
                }
                contrast /= (data.length / 4);

                resolve({
                    width: this.naturalWidth,
                    height: this.naturalHeight,
                    brightness: brightness,
                    contrast: contrast,
                    fileSize: file.size
                });
            };
            img.onerror = function () {
                resolve(null);
            };
            img.src = URL.createObjectURL(file);
        });
    }

    // Enhanced preview with quality info
    fileInput.addEventListener('change', async function (e) {
        const file = e.target.files[0];
        if (file) {
            showPreview(file);

            // Optional: Show quality pre-check
            const qualityInfo = await preCheckImageQuality(file);
            if (qualityInfo) {
                // Add quality indicators to preview
                const qualityIndicator = document.createElement('div');
                qualityIndicator.className = 'quality-indicator';
                qualityIndicator.innerHTML = `
                    <div class="quality-badge ${qualityInfo.brightness < 30 ? 'warning' : qualityInfo.brightness > 220 ? 'warning' : 'good'}">
                        ${qualityInfo.brightness < 30 ? '⚠️ Tối' : qualityInfo.brightness > 220 ? '⚠️ Sáng' : '✅ Sáng tốt'}
                    </div>
                    <div class="quality-badge ${qualityInfo.contrast < 20 ? 'warning' : 'good'}">
                        ${qualityInfo.contrast < 20 ? '⚠️ Thiếu tương phản' : '✅ Tương phản tốt'}
                    </div>
                `;

                // Remove existing quality indicator
                const existing = document.querySelector('.quality-indicator');
                if (existing) {
                    existing.remove();
                }

                // Add new quality indicator
                const previewInfo = document.querySelector('.preview-info');
                previewInfo.appendChild(qualityIndicator);
            }
        }
    });
});

// Add quality indicator styles
const qualityStyles = document.createElement('style');
qualityStyles.textContent = `
    .quality-indicator {
        display: flex;
        gap: 10px;
        margin-top: 10px;
        flex-wrap: wrap;
    }
    
    .quality-badge {
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .quality-badge.good {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .quality-badge.warning {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
`;
document.head.appendChild(qualityStyles);


