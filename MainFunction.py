# MainFunction.py
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
import Const_parameter as CFG
import os
import io

# ----------------------------------------------------
# 1. CẤU HÌNH FLASK
# ----------------------------------------------------
app = Flask(__name__, template_folder='.')


# ----------------------------------------------------
# 2. HÀM XỬ LÝ ẢNH (OPENCV)
# ----------------------------------------------------

def find_contour_by_threshold(image, thresh_value, min_area=1000):
    """Tiền xử lý, áp dụng Closing và tìm contour lớn nhất dựa trên ngưỡng."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)

    # Áp dụng Phép Đóng (Closing)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, CFG.KERNEL_SIZE)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Tìm Contour lớn nhất
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    best_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(best_contour) < min_area:
        return None

    return best_contour


def calculate_standard_parameters(path):
    """Tính toán tâm và kích thước chuẩn của Tem Trắng từ ảnh mẫu."""
    if not os.path.exists(path):
        print(f"LỖI: Không tìm thấy ảnh chuẩn tại đường dẫn: {path}")
        return False

    image = cv2.imread(path)
    if image is None:
        print(f"LỖI: Không thể đọc ảnh chuẩn từ đường dẫn: {path}")
        return False

    # Tìm contour Tem Trắng (Dùng ngưỡng LABEL)
    label_contour = find_contour_by_threshold(image, CFG.THRESH_VALUE_LABEL)

    if label_contour is None:
        print("LỖI: Không tìm thấy TEM TRẮNG trên ảnh chuẩn. Vui lòng kiểm tra THRESH_VALUE_LABEL.")
        return False

    x, y, w, h = cv2.boundingRect(label_contour)

    # LƯU THÔNG SỐ CHUẨN VÀO DICTIONARY
    CFG.STANDARD_PARAMS['label_center_x'] = x + w // 2
    CFG.STANDARD_PARAMS['label_center_y'] = y + h // 2
    CFG.STANDARD_PARAMS['label_width'] = w
    CFG.STANDARD_PARAMS['label_height'] = h

    print(f"ĐÃ CẬP NHẬT THÔNG SỐ CHUẨN: Tâm X={CFG.STANDARD_PARAMS['label_center_x']}, W={w}, H={h}")
    return True


def get_label_and_product_contour(image):
    """Tìm contour tem dán (trắng) và sản phẩm (hồng)"""

    # 1. Tìm Tem Dán (Trắng) - Dùng ngưỡng cao
    label_contour = find_contour_by_threshold(image.copy(), CFG.THRESH_VALUE_LABEL)

    # 2. Tìm Sản Phẩm (Hồng) - Dùng ngưỡng thấp hơn
    product_contour = find_contour_by_threshold(image.copy(), CFG.THRESH_VALUE_PRODUCT, min_area=5000)

    return label_contour, product_contour


def check_label_quality(image, label_contour, product_contour):
    """Kiểm tra logic: Thiếu tem, Lệch tem."""

    # Lấy thông số chuẩn đã được tính toán
    STD = CFG.STANDARD_PARAMS

    # Kiểm tra xem thông số chuẩn đã được tải chưa
    if STD['label_center_x'] is None:
        return "LỖI HỆ THỐNG: CHƯA XÁC ĐỊNH THAM SỐ CHUẨN TỪ ẢNH MẪU", image.copy()

    output_image = image.copy()

    # --- 1. PHÁT HIỆN LỖI THIẾU TEM ---
    if product_contour is not None and label_contour is None:
        # CÓ biên sản phẩm (hồng), nhưng KHÔNG có biên tem (trắng)
        result = "LỖI: TEM THIẾU"
        color = (0, 0, 255)  # Đỏ

        x, y, w, h = cv2.boundingRect(product_contour)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 3)
        return result, output_image

    # --- 2. TEM KHÔNG TÌM THẤY (CẢ SP CŨNG KHÔNG) ---
    if label_contour is None and product_contour is None:
        return "CẢNH BÁO: KHÔNG TÌM THẤY SP & TEM", output_image

    # --- 3. KIỂM TRA LỖI LỆCH/KÍCH THƯỚC (Nếu tìm thấy cả 2) ---
    if label_contour is not None and product_contour is not None:

        # Lấy thông số Tem thực tế
        x_label, y_label, w_label, h_label = cv2.boundingRect(label_contour)
        center_x_label = x_label + w_label // 2
        center_y_label = y_label + h_label // 2

        # --- A. Kiểm tra Vị trí (Lệch) ---
        distance = np.sqrt((center_x_label - STD['label_center_x']) ** 2 +
                           (center_y_label - STD['label_center_y']) ** 2)
        shift_ok = distance < CFG.TOLERANCE_SHIFT

        # --- B. Kiểm tra Kích thước ---
        size_diff_w = abs(w_label - STD['label_width'])
        size_diff_h = abs(h_label - STD['label_height'])
        size_ok = (size_diff_w < STD['label_width'] * CFG.TOLERANCE_SIZE_PCT) and \
                  (size_diff_h < STD['label_height'] * CFG.TOLERANCE_SIZE_PCT)

        # --- C. Quyết định Kết quả ---
        result = "ĐẠT CHUẨN"
        color = (0, 255, 0)  # Xanh lá

        if not shift_ok:
            result = f"LỖI: TEM LỆCH (D: {int(distance)}px)"
            color = (0, 0, 255)  # Đỏ
        elif not size_ok:
            result = "LỖI: SAI KÍCH THƯỚC TEM"
            color = (255, 0, 255)  # Tím

        # Đánh dấu lên ảnh (Vẽ biên SP và Tem)
        x_prod, y_prod, w_prod, h_prod = cv2.boundingRect(product_contour)
        cv2.rectangle(output_image, (x_prod, y_prod), (x_prod + w_prod, y_prod + h_prod), (255, 0, 0),
                      2)  # Biên SP (Xanh dương)
        cv2.rectangle(output_image, (x_label, y_label), (x_label + w_label, y_label + h_label), color, 3)  # Biên Tem
        cv2.circle(output_image, (center_x_label, center_y_label), 5, color, -1)

        return result, output_image

    return "LỖI: TRẠNG THÁI KHÔNG XÁC ĐỊNH", output_image


# ----------------------------------------------------
# 3. ROUTES FLASK VÀ CHƯƠNG TRÌNH CHÍNH
# ----------------------------------------------------

@app.route('/')
def index():
    return render_template('web_test.html')


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image_stream = image_file.read()
    np_array = np.frombuffer(image_stream, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Failed to decode image"}), 400

    # Chạy logic kiểm tra tem dán mới
    label_contour, product_contour = get_label_and_product_contour(image.copy())
    result, output_image = check_label_quality(image, label_contour, product_contour)

    _, buffer = cv2.imencode('.jpg', output_image)
    processed_image_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

    return jsonify({
        "result": result,
        "processed_image_base64": processed_image_base64
    })


@app.route('/upload_and_process', methods=['POST'])
def upload_and_process():
    # Phần này cần được cập nhật logic cho video nếu bạn muốn kiểm tra video
    return "Tính năng xử lý video chưa được cập nhật logic mới...", 501


if __name__ == '__main__':
    print("--- ĐANG TÍNH TOÁN THAM SỐ CHUẨN TỪ ẢNH MẪU ---")
    # KHỞI TẠO VÀ TÍNH TOÁN THAM SỐ CHUẨN KHI CHƯƠNG TRÌNH KHỞI ĐỘNG
    if calculate_standard_parameters(CFG.STANDARD_IMAGE_PATH):
        print("--- THÀNH CÔNG: Sẵn sàng chạy server Flask ---")
        # Dùng use_reloader=False để tránh việc Flask chạy hàm calculate_standard_parameters hai lần
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    else:
        print("--- THẤT BẠI: Vui lòng kiểm tra đường dẫn ảnh chuẩn và THRESH_VALUE_LABEL trong Const_parameter.py ---")