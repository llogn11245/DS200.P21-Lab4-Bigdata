from PIL import Image
import numpy as np

def preprocess_image(image_path: str):
    """
    Đọc ảnh từ đường dẫn, resize về 64x64, chuyển ảnh về mức xám (grayscale),
    chuẩn hóa pixel về [0,1] và trả về danh sách các giá trị đặc trưng (feature).
    """
    # Mở ảnh và chuyển về mức xám
    img = Image.open(image_path).convert('L')
    # Thay đổi kích thước ảnh về 64x64
    img = img.resize((64, 64))
    # Chuyển ảnh thành mảng numpy
    arr = np.array(img, dtype=np.float32)
    # Chuẩn hóa giá trị pixel từ 0-255 về 0-1
    arr = arr / 255.0
    # Chuyển mảng 2D thành vector 1D (flatten) và thành list
    features = arr.flatten().tolist()
    return features
