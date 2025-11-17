
import numpy as np

# ----------------------------------------------------
# 1. Hằng số Mẫu Chuẩn (Standard Constants)
# ----------------------------------------------------
# Vị trí tâm tem chuẩn (pixel)
CENTER_X_STD = 300  
CENTER_Y_STD = 250
# Kích thước tem chuẩn (pixel)
WIDTH_STD = 150
HEIGHT_STD = 80

# ----------------------------------------------------
# 2. Hằng số Ngưỡng Sai số (Tolerance Thresholds)
# ----------------------------------------------------
# Ngưỡng chấp nhận sai số vị trí (pixel)
TOLERANCE_SHIFT = 10 
# Sai số kích thước tối đa (ví dụ: 0.15 = 15%)
TOLERANCE_SIZE_PCT = 0.15 

# ----------------------------------------------------
# 3. Hằng số Xử lý Ảnh (Image Processing)
# ----------------------------------------------------
# Kích thước Kernel cho phép đóng (Closing)
KERNEL_SIZE = (5, 5) 
# Ngưỡng nhị phân hóa (có thể cần điều chỉnh sau)
THRESH_VALUE = 120 

# Kernel cấu trúc cho phép đóng (Structural Element)
CLOSING_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE)