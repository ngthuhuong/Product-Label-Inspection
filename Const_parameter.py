# Const_parameter.py
import cv2
import numpy as np

# --- 0. ĐƯỜNG DẪN ẢNH CHUẨN ---
# BẠN PHẢI THAY ĐỔI ĐƯỜNG DẪN NÀY ĐỂ TRỎ ĐẾN FILE 'chuan.jpg' CỦA BẠN
# Ví dụ: "D:/MyProjects/Label_Inspector/chuan.jpg"
STANDARD_IMAGE_PATH = r"C:\Users\Admin\Documents\_nam 4\huong\xu ly anh\source2\chuan.jpg" # Giả định file chuan.jpg nằm cùng thư mục

# 1. Hằng số Ngưỡng Sai số
TOLERANCE_SHIFT = 30        # Ngưỡng lệch vị trí tối đa (pixel)
TOLERANCE_SIZE_PCT = 0.10  # Sai số kích thước 10%

# 2. Hằng số Xử lý Ảnh
KERNEL_SIZE = (5, 5)
# Ngưỡng cao để tách TEM TRẮNG khỏi nền hồng/gỗ
THRESH_VALUE_LABEL = 200
# Ngưỡng thấp hơn để tách SP HỒNG khỏi nền gỗ
THRESH_VALUE_PRODUCT = 100

# 3. THÔNG SỐ CHUẨN (SẼ ĐƯỢC TÍNH TOÁN KHI CHƯƠNG TRÌNH KHỞI ĐỘNG)
# CHÚ Ý: KHÔNG CẦN NHẬP GIÁ TRỊ VÀO ĐÂY
STANDARD_PARAMS = {
    'label_center_x': None,
    'label_center_y': None,
    'label_width': None,
    'label_height': None
}