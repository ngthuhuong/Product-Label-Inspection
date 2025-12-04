# measure_ratio.py
import cv2
import numpy as np
from pathlib import Path
import config


# ============= HÀM PHỤ GIỐNG main.py =============

def make_top_square(img):
    """Cắt ảnh về hình vuông (lấy phần top)."""
    h, w = img.shape[:2]
    if h == w:
        return img
    if h > w:
        return img[0:w, 0:w]
    else:
        return img[0:h, 0:h]


def norm_angle(a):
    """Chuẩn hoá góc về [-45, 45] giống trong main."""
    if a < -45:
        a += 90
    return a


def find_product_and_label(img):
    """
    Đo các tham số trên 1 ảnh chuẩn.

    Trả về:
        product_area,
        label_area,
        area_ratio,
        offset,
        relative_tilt,
        rectangularity,
        complexity,
        debug_image

    Nếu thất bại: (None, None, None, None, None, None, None, img_square)
    """
    img_square = make_top_square(img)
    frame = img_square.copy()
    debug = frame.copy()

    # =========================
    # 1. TÌM SẢN PHẨM (HỘP VÀNG)
    # =========================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array(config.LOWER_YELLOW)
    upper_yellow = np.array(config.UPPER_YELLOW)

    product_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Morphology (CLOSING + OPENING) cho hộp
    kernel_prod = np.ones(
        (config.KERNEL_PRODUCT_SIZE, config.KERNEL_PRODUCT_SIZE), np.uint8
    )
    product_mask = cv2.morphologyEx(
        product_mask, cv2.MORPH_CLOSE, kernel_prod,
        iterations=config.PRODUCT_CLOSE_ITER
    )
    product_mask = cv2.morphologyEx(
        product_mask, cv2.MORPH_OPEN, kernel_prod,
        iterations=config.PRODUCT_OPEN_ITER
    )

    contours, _ = cv2.findContours(
        product_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("[WARN] Không tìm được contour sản phẩm.")
        return (None, None, None, None, None, None, None, img_square)

    product_cnt = max(contours, key=cv2.contourArea)
    cv2.drawContours(debug, [product_cnt], -1, (0, 255, 0), 2)

    # diện tích hộp: dùng contourArea cho ổn định với góc nghiêng
    product_area = cv2.contourArea(product_cnt)

    # boundingRect để cắt ROI tem
    x, y, w, h = cv2.boundingRect(product_cnt)
    roi = frame[y:y + h, x:x + w]

    # minAreaRect của hộp để lấy góc hộp
    prod_rect = cv2.minAreaRect(product_cnt)
    product_angle = prod_rect[2]
    prod_angle_n = norm_angle(product_angle)
    prod_cx = x + w / 2.0
    prod_cy = y + h / 2.0

    # =========================
    # 2. TÌM TEM TRONG ROI
    # =========================

    # 2.1 Threshold theo độ sáng
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_roi, config.GAUSSIAN_BLUR_KERNEL, 0)
    _, th_bin = cv2.threshold(
        blur, config.THRESH_BINARY_VALUE, 255, cv2.THRESH_BINARY
    )

    # 2.2 Lọc theo màu tem (HSV – gần trắng)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_color = cv2.inRange(
        hsv_roi,
        np.array(config.LABEL_LOWER_HSV),
        np.array(config.LABEL_UPPER_HSV),
    )

    # 2.3 Kết hợp: vừa sáng, vừa đúng màu
    th = cv2.bitwise_and(th_bin, mask_color)

    # 2.4 CLOSING cho tem
    kernel_lbl = np.ones(
        (config.KERNEL_LABEL_SIZE, config.KERNEL_LABEL_SIZE), np.uint8
    )
    th = cv2.morphologyEx(
        th, cv2.MORPH_CLOSE, kernel_lbl,
        iterations=config.LABEL_CLOSE_ITER
    )

    contours_label, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_label = sorted(contours_label, key=cv2.contourArea, reverse=True)

    if not contours_label:
        print("[WARN] Không tìm thấy contour nào trong ROI tem.")
        return (product_area, None, None, None, None, None, None, debug)

    label_area = None
    chosen_cnt = None
    offset = None
    relative_tilt = None
    rectangularity = None
    complexity = None

    for cnt in contours_label:
        area = cv2.contourArea(cnt)

        # 2.5 lọc diện tích cơ bản giống main.py
        if not (
            config.LABEL_MIN_AREA
            < area
            < w * h * config.LABEL_MAX_AREA_RATIO
        ):
            continue

        # 2.6 độ phức tạp contour
        perim = cv2.arcLength(cnt, True)
        complexity_val = perim / (area + 1e-6)
        if complexity_val < config.LABEL_COMPLEXITY_MIN:
            continue
        # nếu dùng MAX thì mở thêm:
        # if complexity_val > config.LABEL_COMPLEXITY_MAX:
        #     continue

        # 2.7 dịch sang toạ độ ảnh gốc & kiểm tra nằm trong hộp
        cnt_shifted = cnt + [x, y]

        M = cv2.moments(cnt_shifted)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        inside = cv2.pointPolygonTest(product_cnt, (cx, cy), False)
        if inside < 0:
            continue

        # 2.8 tính các tham số giống detect_label
        rect_label = cv2.minAreaRect(cnt_shifted)
        (lcx, lcy), (rw, rh), label_angle = rect_label

        # lệch tâm (chuẩn hoá)
        dx = abs(lcx - prod_cx) / w
        dy = abs(lcy - prod_cy) / h
        offset_val = max(dx, dy)

        # lệch góc tương đối
        label_angle_n = norm_angle(label_angle)
        relative_tilt_val = abs(label_angle_n - prod_angle_n)

        # độ vuông vắn
        rect_area = rw * rh if rw * rh > 0 else area
        rectangularity_val = area / rect_area if rect_area > 0 else 0.0

        # lấy contour đầu tiên qua được tất cả filter
        label_area = area
        chosen_cnt = cnt_shifted
        offset = offset_val
        relative_tilt = relative_tilt_val
        rectangularity = rectangularity_val
        complexity = complexity_val
        break

    if chosen_cnt is None:
        print("[WARN] Không có contour tem nào qua được filter.")
        return (product_area, None, None, None, None, None, None, debug)

    # vẽ tem lên debug
    cv2.drawContours(debug, [chosen_cnt], -1, (0, 0, 255), 2)
    lx, ly, lw, lh = cv2.boundingRect(chosen_cnt)
    cv2.putText(
        debug,
        "Tem (calib)",
        (lx, ly - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )

    area_ratio = label_area / float(product_area)

    return (
        product_area,
        label_area,
        area_ratio,
        offset,
        relative_tilt,
        rectangularity,
        complexity,
        debug,
    )


def main():
    # ĐỔI TÊN FILE Ở ĐÂY ĐỂ TEST ẢNH KHÁC
    IMAGE_PATH = Path(
        r"C:\Users\Admin\Documents\_nam 4\huong\xu ly anh\Label_Inspector\1.jpg"
    )

    img = cv2.imread(str(IMAGE_PATH))
    if img is None:
        print(f"[ERROR] Không đọc được ảnh: {IMAGE_PATH}")
        return

    (
        product_area,
        label_area,
        area_ratio,
        offset,
        relative_tilt,
        rectangularity,
        complexity,
        debug,
    ) = find_product_and_label(img)

    print("========== KẾT QUẢ CALIBRATION ==========")
    print(f"Ảnh: {IMAGE_PATH}")

    if (
        product_area is None
        or label_area is None
        or area_ratio is None
    ):
        print("❌ Không đo được tỉ lệ tem / sản phẩm.")
    else:
        print(f"- Diện tích hộp (product_area)      = {product_area}")
        print(f"- Diện tích tem (label_area)        = {label_area}")
        print(f"- TỈ LỆ tem/hộp (area_ratio)        = {area_ratio:.6f}")
        if offset is not None:
            print(f"- OFFSET (lệch tâm, max(dx,dy))    = {offset:.6f}")
        if relative_tilt is not None:
            print(f"- GÓC LỆCH TƯƠNG ĐỐI (độ)          = {relative_tilt:.3f}")
        if rectangularity is not None:
            print(f"- RECTANGULARITY (area/rect_area)  = {rectangularity:.6f}")
        if complexity is not None:
            print(f"- COMPLEXITY (perimeter/area)      = {complexity:.6f}")
        print()
        print("👉 Gợi ý cấu hình:")
        print(f"  LABEL_AREA_RATIO_REF = {area_ratio:.6f}")
        print("  LABEL_AREA_RATIO_TOL = 0.01  # thử ±0.01 rồi điều chỉnh thêm")

    # Hiển thị ảnh debug
    cv2.imshow("Calibration Debug", debug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
