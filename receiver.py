from trainer import SparkConfig, Trainer
from models.model import Model
import argparse

# parser = argparse.ArgumentParser(description="Receiver - Nhận dữ liệu và huấn luyện mô hình")
# parser.add_argument('--model', '-m', required=True, type=str, help='Tên mô hình')

if __name__ == "__main__":
    # args = parser.parse_args()
    # model_name = args.model

    # Khởi tạo cấu hình Spark
    spark_conf = SparkConfig()
    # Lựa chọn thuật toán mô hình (có thể đặt trong SparkConfig); ở đây dùng logistic regression
    spark_conf.algorithm = "logistic"

    # Khởi tạo mô hình và Trainer
    model = Model(algorithm=spark_conf.algorithm)
    trainer = Trainer(model, spark_conf)

    # Bắt đầu huấn luyện trên luồng dữ liệu streaming
    # Lưu ý: Chạy file này bằng Spark (ví dụ: spark-submit receiver.py)
    trainer.start_training()
