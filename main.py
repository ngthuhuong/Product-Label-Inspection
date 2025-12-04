from flask import Flask, request, jsonify, send_file, Response
import cv2
import numpy as np
import base64
import time

# Import tất cả tham số
import config

app = Flask(__name__)

current_status = "khong_tim_thay"


# ==========================
# HÀM HỖ TRỢ
# ==========================

def b64_to_image(b64_str):
    img_bytes = base64.b64decode(b64_str)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def make_top_square(img):
    """Cắt ảnh về hình vuông (lấy phần top)."""
    h, w = img.shape[:2]
    if h == w:
        return img
    if h > w:
        return img[0:w, 0:w]
    else:
        return img[0:h, 0:h]


# ==========================
# PHÂN LOẠI TEM
# ==========================

def classify_label(product_rect, product_angle, label_cnt):
    px, py, pw, ph = product_rect
    lx, ly, lw, lh = cv2.boundingRect(label_cnt)

    rect = cv2.minAreaRect(label_cnt)
    (cx, cy), (rw, rh), label_angle = rect

    def norm_angle(a):
        if a < -45:
            a += 90
        return a

    prod_angle_n = norm_angle(product_angle)
    label_angle_n = norm_angle(label_angle)

    # 1) lệch tâm
    prod_cx = px + pw / 2.0
    prod_cy = py + ph / 2.0
    dx = abs(cx - prod_cx) / pw
    dy = abs(cy - prod_cy) / ph
    offset = max(dx, dy)

    # 2) lệch góc tương đối
    relative_tilt = abs(label_angle_n - prod_angle_n)

    # 3) độ vuông vắn
    contour_area = cv2.contourArea(label_cnt)
    rect_area = rw * rh if rw * rh > 0 else lw * lh
    rectangularity = contour_area / rect_area if rect_area > 0 else 0

    # ----- PHÂN LOẠI -----
    if (rectangularity >= config.RECT_GOOD and
            offset <= config.OFF_CENTER_OK and
            relative_tilt <= config.REL_TILT_OK):
        return "dan_dung"
    elif rectangularity >= config.RECT_OK:
        return "dan_lech"
    else:
        return "tem_rach"


# ==========================
# NHẬN DIỆN TEM & SẢN PHẨM
# ==========================

def detect_label(img):
    # 0. Cắt ảnh về hình vuông (lấy phần top)
    img_square = make_top_square(img)
    frame = img_square.copy()
    output = frame.copy()

    # =========================
    # 1. TÌM SẢN PHẨM (HỘP VÀNG)
    # =========================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array(config.LOWER_YELLOW)
    upper_yellow = np.array(config.UPPER_YELLOW)

    product_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 🔒 CLOSING + OPENING cho hộp (BẮT BUỘC)
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
        # không thấy hộp → trả về như cũ
        _, orig_buf = cv2.imencode(".jpg", img_square)
        _, mask_buf = cv2.imencode(".jpg", product_mask)
        _, out_buf  = cv2.imencode(".jpg", output)
        return (
            base64.b64encode(mask_buf).decode("utf-8"),
            base64.b64encode(out_buf).decode("utf-8"),
            "khong_tim_thay",
            base64.b64encode(orig_buf).decode("utf-8"),
        )

    # contour hộp lớn nhất
    product_cnt = max(contours, key=cv2.contourArea)
    prod_rect = cv2.minAreaRect(product_cnt)   # ((cx,cy),(w,h),angle)
    product_angle = prod_rect[2]

    # ❗Dùng diện tích CONTOUR để tránh phụ thuộc góc nghiêng
    product_area = cv2.contourArea(product_cnt)

    # vẽ viền hộp
    cv2.drawContours(output, [product_cnt], -1, (0, 255, 0), 3)

    # bounding box hộp để cắt ROI tìm tem
    x, y, w, h = cv2.boundingRect(product_cnt)
    roi = frame[y:y + h, x:x + w]

    # =========================
    # 2. TÌM TEM TRONG ROI
    # =========================
    # --- TÌM TEM TRONG ROI ---

    # 1) Threshold theo độ sáng
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_roi, config.GAUSSIAN_BLUR_KERNEL, 0)
    _, th_bin = cv2.threshold(
        blur, config.THRESH_BINARY_VALUE, 255, cv2.THRESH_BINARY
    )

    # 2) Lọc theo màu tem (HSV gần trắng)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_color = cv2.inRange(
        hsv_roi,
        np.array(config.LABEL_LOWER_HSV),
        np.array(config.LABEL_UPPER_HSV)
    )

    # 3) Kết hợp: chỉ giữ vùng vừa sáng vừa đúng màu
    th = cv2.bitwise_and(th_bin, mask_color)

    # 4) CLOSING (bắt buộc để liền nét)
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

    status = "khong_tim_thay"

    # =========================
    # 3. DUYỆT CÁC CONTOUR TEM
    # =========================
    for cnt in contours_label:
        area = cv2.contourArea(cnt)

        # 3.1 lọc diện tích cơ bản
        if not (
            config.LABEL_MIN_AREA
            < area
            < w * h * config.LABEL_MAX_AREA_RATIO
        ):
            continue

        # 3.2 độ phức tạp contour (lọc nền trắng phẳng)
        perim = cv2.arcLength(cnt, True)
        complexity = perim / (area + 1e-6)
        if complexity < config.LABEL_COMPLEXITY_MIN:
            continue
        # nếu muốn dùng max thì mở lại:
        # if complexity > config.LABEL_COMPLEXITY_MAX:
        #     continue

        # 3.3 chuyển sang tọa độ ảnh gốc
        cnt_shifted = cnt + [x, y]

        # 3.4 tâm contour phải nằm trong hộp vàng
        M = cv2.moments(cnt_shifted)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        inside = cv2.pointPolygonTest(product_cnt, (cx, cy), False)
        if inside < 0:
            continue

        # =========================
        # 4. TÍNH CÁC THÔNG SỐ
        # =========================
        rect_label = cv2.minAreaRect(cnt_shifted)   # ((cx,cy),(rw,rh),angle)
        (lcx, lcy), (rw, rh), label_angle = rect_label

        # lệch tâm (chuẩn hóa theo kích thước hộp)
        prod_cx = x + w / 2.0
        prod_cy = y + h / 2.0
        dx = abs(lcx - prod_cx) / w
        dy = abs(lcy - prod_cy) / h
        offset = max(dx, dy)

        # chuẩn hóa góc về [-45,45]
        def norm_angle(a):
            if a < -45:
                a += 90
            return a

        prod_angle_n  = norm_angle(product_angle)
        label_angle_n = norm_angle(label_angle)
        relative_tilt = abs(label_angle_n - prod_angle_n)

        # độ vuông vắn tem
        rect_area = rw * rh if rw * rh > 0 else area
        rectangularity = area / rect_area if rect_area > 0 else 0.0

        # tỉ lệ diện tích tem / hộp (dùng contour area của hộp)
        area_ratio = area / float(product_area)

        # =========================
        # 5. CỜ LỖI LỆCH / RÁCH
        # =========================
        is_lech = (
            offset > config.OFF_CENTER_OK or
            relative_tilt > config.REL_TILT_OK
        )

        is_rach = (
            area_ratio < config.LABEL_AREA_RATIO_REF - config.LABEL_AREA_RATIO_TOL
            or rectangularity < config.RECT_OK
        )

        # =========================
        # 6. SUY RA TRẠNG THÁI
        # =========================
        if (not is_lech) and (not is_rach):
            status = "dan_dung"
            color = (0, 255, 0)
            text = "Tem OK"
        elif is_lech and (not is_rach):
            status = "dan_lech"
            color = (0, 255, 255)
            text = "Tem lech"
        elif (not is_lech) and is_rach:
            status = "tem_rach"
            color = (0, 0, 255)
            text = "Tem rach"
        else:
            status = "tem_lech_rach"
            color = (0, 128, 255)   # cam
            text = "Lech + rach"

        # vẽ contour + text
        cv2.drawContours(output, [cnt_shifted], -1, color, 2)
        lx, ly, lw, lh = cv2.boundingRect(cnt_shifted)
        cv2.putText(
            output,
            text,
            (lx, ly - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        break  # chỉ lấy tem hợp lệ đầu tiên

    # nếu vẫn không tìm được tem
    if status == "khong_tim_thay":
        cv2.putText(
            output,
            "Khong co tem",
            (x + 10, y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

    # =========================
    # 7. ENCODE TRẢ VỀ WEB
    # =========================
    _, mask_buf  = cv2.imencode(".jpg", product_mask)
    _, out_buf   = cv2.imencode(".jpg", output)
    _, orig_buf  = cv2.imencode(".jpg", img_square)

    return (
        base64.b64encode(mask_buf).decode("utf-8"),
        base64.b64encode(out_buf).decode("utf-8"),
        status,
        base64.b64encode(orig_buf).decode("utf-8"),
    )



# ==========================
# REALTIME STREAM
# ==========================

def gen_frames():
    global current_status, cam

    prev_gray = None
    stable_counter = 0
    last_detect_time = 0
    last_result_img = None

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            result_img = frame.copy()
        else:
            diff = cv2.absdiff(gray, prev_gray)
            _, diff_th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            motion_pixels = cv2.countNonZero(diff_th)
            total_pixels = diff_th.size
            motion_ratio = motion_pixels / float(total_pixels)

            prev_gray = gray

            # motion detection
            if motion_ratio < config.MOTION_THRESHOLD:
                stable_counter += 1
            else:
                stable_counter = 0

            now = time.time()
            should_detect = (
                stable_counter >= config.STABLE_FRAMES
                and (now - last_detect_time) >= config.MIN_DETECT_GAP
            )

            if should_detect:
                mask_b64, output_b64, status, origin_b64 = detect_label(frame)
                current_status = status
                last_detect_time = now
                result_img = b64_to_image(output_b64)
                last_result_img = result_img.copy()
            else:
                if last_result_img is not None:
                    result_img = last_result_img.copy()
                else:
                    result_img = frame.copy()

        ret2, buffer = cv2.imencode(".jpg", result_img)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


# ==========================
# ROUTES
# ==========================

@app.route("/")
def index():
    return send_file("new_index.html")


@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "No image uploaded"}), 400

    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    mask_b64, output_b64, status, origin_b64 = detect_label(img)

    return jsonify(
        {
            "mask": mask_b64,
            "result": output_b64,
            "status": status,
            "origin": origin_b64,
        }
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/rt_status")
def rt_status():
    return jsonify({"status": current_status})


@app.route("/realtime")
def realtime_page():
    return send_file("realtime.html")


if __name__ == "__main__":
    # Khởi tạo camera
    cam = cv2.VideoCapture(config.CAM_INDEX)
    app.run(host="0.0.0.0", port=5000, debug=True)
