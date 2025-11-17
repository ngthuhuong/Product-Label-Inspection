import cv2
import numpy as np
import matplotlib.pyplot as plt
import Const_parameter as const
import os

def show_image(title, image):
    """Hiển thị ảnh bằng Matplotlib"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


def preprocess_and_find_label(image):
    """
    Thực hiện tiền xử lý, phép đóng và tìm đường viền tem.
    """
    # 1. Chuyển sang ảnh Xám và Lọc nhiễu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Nhị phân hóa (Tách tem khỏi nền)
    _, binary = cv2.threshold(blurred, const.THRESH_VALUE, 255, cv2.THRESH_BINARY)
    #

    # 3. Áp dụng Phép Đóng (Closing) - Yêu cầu 40.1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, const.KERNEL_SIZE)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    #

    # 4. Tìm Đường viền (Contours)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Lọc và chọn contour lớn nhất/phù hợp nhất
    if not contours:
        return None, None  # Không tìm thấy tem

    # Chọn contour có diện tích lớn nhất (giả định là tem)
    # (Có thể cần lọc kỹ hơn bằng tỉ lệ khung hình thực tế)
    best_contour = max(contours, key=cv2.contourArea)

    return best_contour, closed


def check_label_quality(image, contour):
    """
    Kiểm tra vị trí, kích thước và báo cáo chất lượng tem.
    """
    if contour is None:
        return "LỖI: TEM THIẾU", image

    # Lấy Khung bao (Bounding Box)
    x, y, w, h = cv2.boundingRect(contour)

    # 1. Kiểm tra Kích thước
    size_ok = (abs(w - const.WIDTH_STD) < const.WIDTH_STD * const.TOLERANCE_SIZE_PCT) and \
              (abs(h - const.HEIGHT_STD) < const.HEIGHT_STD * const.TOLERANCE_SIZE_PCT)

    # 2. Kiểm tra Vị trí (Tâm tem)
    center_x = x + w // 2
    center_y = y + h // 2
    distance = np.sqrt((center_x - const.CENTER_X_STD) ** 2 + (center_y - const.CENTER_Y_STD) ** 2)
    shift_ok = distance < const.TOLERANCE_SHIFT

    # 3. Tổng hợp kết quả và Đánh dấu
    result = "ĐẠT CHUẨN"
    color = (0, 255, 0)  # Xanh lá

    if not shift_ok:
        result = "LỖI: TEM LỆCH"
        color = (255, 0, 0)  # Đỏ
    elif not size_ok:
        result = "LỖI: SAI KÍCH THƯỚC"
        color = (255, 0, 255)  # Hồng
    # (Thêm logic kiểm tra tem rách/hỏng ở đây, ví dụ: kiểm tra độ lồi)

    # Vẽ khung và tâm tem lên ảnh
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
    cv2.circle(image, (center_x, center_y), 5, color, -1)

    return result, image