# ===============================
# CONFIG THAM SỐ DỰ ÁN NHẬN DIỆN TEM
# ===============================

# -----------------------------------
# 1) Ngưỡng nhận diện MÀU SẢN PHẨM (HSV)
# -----------------------------------
LOWER_YELLOW = (8, 40, 40)
UPPER_YELLOW = (35, 255, 255)
LABEL_LOWER_HSV = (0, 0, 180)      # H bất kỳ, S thấp, V cao
LABEL_UPPER_HSV = (179, 60, 255)   # cho phép hơi xám / hơi màu

# Morphology cho SẢN PHẨM
KERNEL_PRODUCT_SIZE = 7
PRODUCT_CLOSE_ITER = 2
PRODUCT_OPEN_ITER = 1


# -----------------------------------
# 2) Ngưỡng nhận diện TEM (trong ROI)
# -----------------------------------
GAUSSIAN_BLUR_KERNEL = (5, 5)
THRESH_BINARY_VALUE = 200

# Morphology cho TEM
KERNEL_LABEL_SIZE = 3
LABEL_CLOSE_ITER = 2

# Diện tích tem hợp lệ
LABEL_MIN_AREA = 1500
LABEL_MAX_AREA_RATIO = 0.9   # < 90% hộp => không bị dính cả sản phẩm


# -----------------------------------
# 3) Ngưỡng phân loại TEM
# -----------------------------------
OFF_CENTER_OK = 0.06          # lệch tâm < 6% kích thước hộp
REL_TILT_OK = 15.0             # lệch góc < 3 độ
RECT_GOOD = 0.90              # tem chuẩn
RECT_OK = 0.90                # tem lệch

# < 0.9 coi là rách
# Diện tích tem so với sản phẩm (tỉ lệ)
# r_ref ≈ diện tích tem / diện tích hộp ở ảnh chuẩn 1.jpg
LABEL_AREA_RATIO_REF = 0.23    # ví dụ, bạn sẽ chỉnh lại cho đúng thực tế
LABEL_AREA_RATIO_TOL = 0.02   # cho phép lệch ±2%

# -----------------------------------
# 6) Độ phức tạp contour của TEM( trường hợp nhầm với nền trắng phẳng)
#    complexity = perimeter / area
#    Tem thật có complexity lớn hơn nền trắng phẳng
# -----------------------------------
LABEL_COMPLEXITY_MIN = 0.005   # hoặc 0.008, đừng để to quá
LABEL_COMPLEXITY_MAX = 0.20    # hiện tại đang không dùng, nhưng cứ để rộng



# -----------------------------------
# 4) Real-time: kiểm tra chuyển động
# -----------------------------------
MOTION_THRESHOLD = 0.02        # < 2% pixel thay đổi => đứng yên
STABLE_FRAMES = 5              # cần đứng yên liên tiếp
MIN_DETECT_GAP = 0.5           # giãn cách giữa 2 detect (giây)


# -----------------------------------
# 5) Camera settings
# -----------------------------------
CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_LIMIT = None              # có thể đặt 10, 15... để giảm load
