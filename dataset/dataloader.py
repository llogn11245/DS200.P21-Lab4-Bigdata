import os
import random

class ImageDataLoader:
    """
    Lớp hỗ trợ tải danh sách file ảnh và tạo batch dữ liệu ảnh.
    """
    def __init__(self, folder: str, batch_size: int):
        """
        Khởi tạo với thư mục chứa ảnh và kích thước batch.
        """
        self.folder = folder
        self.batch_size = batch_size
        self.images_list = []  # Danh sách các (đường dẫn ảnh, nhãn)
        self.index = 0         # Chỉ mục hiện tại trong danh sách ảnh
        self._load_images()
        # Xáo trộn thứ tự ảnh để dữ liệu đa dạng hơn (tuỳ chọn)
        random.shuffle(self.images_list)

    def _load_images(self):
        """
        Duyệt qua thư mục và thu thập tất cả các file ảnh cùng nhãn (chó=0, mèo=1).
        """
        for root, dirs, files in os.walk(self.folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Xác định nhãn dựa vào tên file hoặc tên thư mục (chó=0, mèo=1)
                    path = os.path.join(root, file)
                    label = None
                    if "dog" in root.lower():
                        label = 0  # ảnh chó
                    elif "cat" in root.lower():
                        label = 1  # ảnh mèo
                    # Bỏ qua nếu không xác định được nhãn
                    if label is None:
                        continue
                    # Thêm vào danh sách ảnh
                    self.images_list.append((path, label))

    def get_batch(self):
        """
        Lấy một batch ảnh tiếp theo (danh sách các (path, label) có độ dài batch_size).
        Nếu đến cuối danh sách thì quay vòng lại từ đầu (tiếp tục streaming vô hạn).
        """
        if len(self.images_list) == 0:
            return None
        # Nếu còn đủ ảnh cho batch đầy đủ
        if self.index + self.batch_size <= len(self.images_list):
            batch = self.images_list[self.index : self.index + self.batch_size]
            self.index += self.batch_size
        else:
            # Nếu ảnh còn lại không đủ một batch, lấy phần cuối và phần đầu danh sách
            batch = self.images_list[self.index:]  # lấy hết ảnh còn lại
            remaining = self.batch_size - len(batch)  # số ảnh cần thêm cho đủ batch
            if remaining > 0:
                # Lấy thêm ảnh từ đầu danh sách để đủ batch
                batch += self.images_list[:remaining]
            # Bắt đầu một vòng mới từ vị trí 'remaining'
            self.index = remaining
        return batch
