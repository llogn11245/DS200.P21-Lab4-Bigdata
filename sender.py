import socket
import json
import time
import argparse

from dataset.dataloader import ImageDataLoader
from transform.transform import preprocess_image

parser = argparse.ArgumentParser(description="Sender - Gửi ảnh chó/mèo qua socket")
parser.add_argument('--folder_path', '-f', required=True, type=str, help='Thư mục chứa ảnh chó và mèo')
parser.add_argument('--batch_size', '-b', required=True, type=int, help='Số lượng ảnh mỗi batch gửi đi')
parser.add_argument('--interval', '-i', required=False, type=int, default=2, help='Khoảng thời gian (giây) giữa hai lần gửi batch')

HOST = "localhost"
PORT = 6100

if __name__ == "__main__":
    args = parser.parse_args()

    IMAGE_FOLDER = args.folder_path       # Thư mục ảnh đầu vào
    BATCH_SIZE = args.batch_size     # Kích thước batch gửi mỗi lần
    SEND_INTERVAL = args.interval    # Thời gian chờ giữa các lần gửi batch

    # Khởi tạo DataLoader để lấy batch ảnh
    data_loader = ImageDataLoader(IMAGE_FOLDER, BATCH_SIZE)

    # Thiết lập socket TCP server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Cho phép tái sử dụng địa chỉ socket ngay sau khi kết thúc (tránh lỗi "Address already in use")
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, PORT))
    sock.listen(1)
    print(f"[Sender] Waiting to connect to port {PORT} ...")
    conn, addr = sock.accept()
    print(f"[Sender] Connected to address {addr}")

    try:
        while True:
            batch = data_loader.get_batch()
            if batch is None:
                print("[Sender] No image in stock.")
                break
            # Chuẩn bị payload JSON cho batch hiện tại
            batch_data = []
            for image_path, label in batch:
                features = preprocess_image(image_path)
                batch_data.append({"X": features, "y": label})
            # Chuyển đổi batch_data thành chuỗi JSON và gửi qua socket
            message = json.dumps(batch_data) + "\n"  # thêm ký tự xuống dòng để consumer phân tách
            try:
                conn.send(message.encode('utf-8'))
                print(f"[Sender] Sent {len(batch_data)} batch of image.")
            except BrokenPipeError:
                print("[Sender] Connection terminated. Stop sending images")
                break
            # Tạm dừng SEND_INTERVAL giây trước khi gửi batch tiếp theo
            time.sleep(SEND_INTERVAL)
    finally:
        # Đóng kết nối socket khi kết thúc
        conn.close()
        sock.close()
        print("[Sender] Connection stop.")
