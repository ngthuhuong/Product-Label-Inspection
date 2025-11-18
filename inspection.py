import cv2
import numpy as np
import matplotlib.pyplot as plt
import Const_parameter as CFG

IMAGE_PATH = r"C:\Users\Admin\Documents\_nam 4\huong\xu ly anh\source\chuan.jpg"


def find_contour_by_threshold(image, thresh_value):
    """Tiền xử lý, áp dụng Closing và tìm contour lớn nhất dựa trên ngưỡng."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Nhị phân hóa
    _, binary = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)

    # Áp dụng Phép Đóng (Closing)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, CFG.KERNEL_SIZE)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Tìm Contour lớn nhất
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Tìm contour lớn nhất theo diện tích
    best_contour = max(contours, key=cv2.contourArea)

    # Lọc contour quá nhỏ (có thể thêm logic lọc kích thước tại đây)
    if cv2.contourArea(best_contour) < 1000:
        return None

    return best_contour


def get_label_and_product_contour(image):
    """Tìm contour tem dán (trắng) và sản phẩm (hồng)"""

    # 1. Tìm Tem Dán (Tem Trắng: THRESH_VALUE_LABEL cao)
    label_contour = find_contour_by_threshold(image.copy(), CFG.THRESH_VALUE_LABEL)

    # 2. Tìm Sản Phẩm (Sản phẩm Hồng: THRESH_VALUE_PRODUCT thấp hơn)
    # Vì SP màu hồng, ta có thể đảo ngược ảnh để tách nó khỏi nền gỗ nếu cần,
    # nhưng thường THRESH_VALUE phù hợp sẽ làm được điều đó.
    product_contour = find_contour_by_threshold(image.copy(), CFG.THRESH_VALUE_PRODUCT)

    return label_contour, product_contour
# --- CHƯƠNG TRÌNH CHÍNH ---
def main_inspection(path):
    # 1. Đọc ảnh
    image = cv2.imread(path)
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh từ {path}")
        return

    # 2. Xử lý và tìm contour
    contour, processed_image = preprocess_and_find_label(image.copy())

    # 3. Kiểm tra chất lượng và nhận kết quả
    result, output_image = check_label_quality(image, contour)

    # 4. Hiển thị kết quả
    print("\n==============================================")
    print(f"KẾT QUẢ KIỂM TRA: {result}")
    print("==============================================")

    # Hiển thị ảnh kết quả
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Kết quả: {result}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main_inspection(IMAGE_PATH)