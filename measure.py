# measure_ratio.py
import cv2
import numpy as np
import config
from pathlib import Path

# ============= HÀM PHỤ DÙNG GIỐNG main.py =============

def make_top_square(img):
    """Cắt ảnh về hình vuông (lấy phần top)."""
    h, w = img.shape[:2]
    if h == w:
        return img
    if h > w:
        return img[0:w, 0:w]
    else:
        return img[0:h, 0:h]


def find_product_and_label(img):
    """
    Trả về:
        product_area, label_area, area_ratio, debug_image
    hoặc (None, None, None, img) nếu không tìm được.
    """
    img_square = make_top_square(img)
    frame = img_square.copy()
    debug = frame.copy()

    # ---- 1. Tìm SẢN PHẨM theo HSV (giống main.py) ----
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array(config.LOWER_YELLOW)
    upper_yellow = np.array(config.UPPER_YELLOW)

    product_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel_prod = np.ones(
        (config.KERNEL_PRODUCT_SIZE, config.KERNEL_PRODUCT_SIZE), np.uint8
    )
    product_mask = cv2.morphologyEx(
        product_mask, cv2.MORPH_CLOSE, kernel_prod, iterations=config.PRODUCT_CLOSE_ITER
    )
    product_mask = cv2.morphologyEx(
        product_mask, cv2.MORPH_OPEN, kernel_prod, iterations=config.PRODUCT_OPEN_ITER
    )

    contours, _ = cv2.findContours(
        product_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("[WARN] Không tìm được contour sản phẩm.")
        return None, None, None, img_square

    product_cnt = max(contours, key=cv2.contourArea)
    cv2.drawContours(debug, [product_cnt], -1, (0, 255, 0), 2)

    x, y, w, h = cv2.boundingRect(product_cnt)
    product_area = w * h

    roi = frame[y:y + h, x:x + w]

    # ---- 2. Tìm TEM trong ROI (giống detect_label) ----
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_roi, config.GAUSSIAN_BLUR_KERNEL, 0)

    _, th = cv2.threshold(
        blur, config.THRESH_BINARY_VALUE, 255, cv2.THRESH_BINARY
    )

    kernel_lbl = np.ones(
        (config.KERNEL_LABEL_SIZE, config.KERNEL_LABEL_SIZE), np.uint8
    )
    th = cv2.morphologyEx(
        th, cv2.MORPH_CLOSE, kernel_lbl, iterations=config.LABEL_CLOSE_ITER
    )

    contours_label, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_label = sorted(contours_label, key=cv2.contourArea, reverse=True)

    if not contours_label:
        print("[WARN] Không tìm thấy contour nào trong ROI tem.")
        return product_area, None, None, debug

    label_area = None
    chosen_cnt = None

    for cnt in contours_label:
        area = cv2.contourArea(cnt)

        # Lọc diện tích cơ bản giống main.py
        if not (
            config.LABEL_MIN_AREA
            < area
            < w * h * config.LABEL_MAX_AREA_RATIO
        ):
            continue

        # Tính complexity như trong main.py
        perim = cv2.arcLength(cnt, True)
        complexity = perim / (area + 1e-6)
        if complexity < config.LABEL_COMPLEXITY_MIN:
            continue
        # nếu bạn có dùng MAX thì thêm vào đây:
        # if complexity > config.LABEL_COMPLEXITY_MAX:
        #     continue

        # Dịch sang tọa độ ảnh gốc để kiểm tra nằm trong hộp
        cnt_shifted = cnt + [x, y]
        M = cv2.moments(cnt_shifted)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        inside = cv2.pointPolygonTest(product_cnt, (cx, cy), False)
        if inside < 0:
            continue

        # Nếu qua được hết filter -> lấy contour này làm tem chuẩn
        label_area = area
        chosen_cnt = cnt_shifted
        break

    if chosen_cnt is None:
        print("[WARN] Không có contour tem nào qua được filter.")
        return product_area, None, None, debug

    # Vẽ tem lên debug image
    cv2.drawContours(debug, [chosen_cnt], -1, (0, 0, 255), 2)
    lx, ly, lw, lh = cv2.boundingRect(chosen_cnt)
    cv2.putText(debug, "Tem (calib)", (lx, ly - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    area_ratio = label_area / float(product_area)

    return product_area, label_area, area_ratio, debug


def main():
    # ĐỔI TÊN FILE Ở ĐÂY ĐỂ TEST ẢNH KHÁC
    IMAGE_PATH = Path(r"C:\Users\Admin\Documents\_nam 4\huong\xu ly anh\phat hien tem\AnhSP\nghieng_nhieu.jpg")

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"[ERROR] Không đọc được ảnh: {IMAGE_PATH}")
        return

    product_area, label_area, area_ratio, debug = find_product_and_label(img)

    print("========== KẾT QUẢ CALIBRATION ==========")
    print(f"Ảnh: {IMAGE_PATH}")

    if product_area is None or label_area is None or area_ratio is None:
        print("❌ Không đo được tỉ lệ tem / sản phẩm.")
    else:
        print(f"- Diện tích hộp (product_area) = {product_area}")
        print(f"- Diện tích tem (label_area)   = {label_area}")
        print(f"- TỈ LỆ tem/hộp (area_ratio)   = {area_ratio:.6f}")
        print()
        print("👉 Gợi ý cấu hình:")
        print(f"   LABEL_AREA_RATIO_REF = {area_ratio:.6f}")
        print("   LABEL_AREA_RATIO_TOL = 0.01  # thử ±0.01 rồi điều chỉnh thêm")

    # Hiển thị ảnh debug cho dễ kiểm tra
    cv2.imshow("Calibration Debug", debug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
