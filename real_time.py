import cv2
import numpy as np
import base64
from main import detect_label  # như cũ

def b64_to_image(b64_str):
    img_bytes = base64.b64decode(b64_str)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def main():
    cam_index = 0
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Không mở được camera iVCam. Kiểm tra lại kết nối & index.")
        return

    print("✅ Đang nhận video từ iVCam... bấm 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Mất frame từ camera.")
            break

        try:
            mask_b64, output_b64, status, origin_b64 = detect_label(frame)
        except Exception as e:
            print("Lỗi detect_label:", e)
            continue

        result_img = b64_to_image(output_b64)

        cv2.imshow("KIỂM TRA TEM REAL-TIME", result_img)
        print("Status:", status, end="\r")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
