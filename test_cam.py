import cv2
import numpy as np

# -----------------------------
# HÀM PREWITT GỐC
# -----------------------------
def prewitt_edges(gray, thresh=None):
    # kernel Prewitt
    kx = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)

    ky = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float32)

    gx = cv2.filter2D(gray, cv2.CV_32F, kx)
    gy = cv2.filter2D(gray, cv2.CV_32F, ky)

    # độ lớn gradient
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag = mag.astype(np.uint8)

    edges = mag.copy()
    if thresh is not None:
        _, edges = cv2.threshold(mag, thresh, 255, cv2.THRESH_BINARY)

    return gx, gy, mag, edges


# -----------------------------
# HÀM TẠO MASK TEM BẰNG PREWITT
# -----------------------------
def build_label_mask_prewitt(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_roi, (5, 5), 0)

    gx, gy, mag, edges = prewitt_edges(blur, thresh=40)

    # giống logic của bạn: closing để liền khối
    kernel_lbl = np.ones((5, 5), np.uint8)  # thay cho config.KERNEL_LABEL_SIZE
    edges_close = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, kernel_lbl, iterations=2
    )

    return gray_roi, blur, gx, gy, mag, edges, edges_close


# -----------------------------
# DEMO TỪNG BƯỚC
# -----------------------------
def main():
    img = cv2.imread("1.jpg")   # đổi path nếu cần
    if img is None:
        print("Không đọc được ảnh 1.jpg")
        return

    # ở đây mình dùng cả ảnh làm ROI, bạn có thể cắt roi riêng tùy bài
    gray, blur, gx, gy, mag, edges, edges_close = build_label_mask_prewitt(img)

    # hiển thị từng bước
    cv2.imshow("1) Original", img)
    cv2.imshow("2) Gray", gray)
    cv2.imshow("3) Blur", blur)

    # convert gx, gy để nhìn được
    gx_vis = cv2.convertScaleAbs(gx)
    gy_vis = cv2.convertScaleAbs(gy)
    cv2.imshow("4) Prewitt Gx", gx_vis)
    cv2.imshow("5) Prewitt Gy", gy_vis)

    cv2.imshow("6) Magnitude (normalized 0-255)", mag)
    cv2.imshow("7) Edges (threshold=40)", edges)
    cv2.imshow("8) Edges after CLOSE", edges_close)

    print("Nhấn phím bất kỳ để thoát…")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
