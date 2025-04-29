from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
import json

from models.model import Model

class SparkConfig:
    """
    Cấu hình cho Spark Streaming.
    """
    appName = "ImageClassifier"   # Tên ứng dụng Spark
    master = "local"             # Chế độ chạy (local)
    num_threads = 2             # Số luồng (threads) cho Spark (2 để đủ cho streaming)
    hostname = "localhost"      # Địa chỉ host của socket (producer)
    port = 6100                 # Cổng của socket (phải trùng với producer)
    batch_interval = 2          # Khoảng thời gian micro-batch (giây)
    algorithm = "logistic"      # Thuật toán mô hình: "logistic", "decision_tree", hoặc "random_forest"

class Trainer:
    def __init__(self, model: Model, spark_config: SparkConfig):
        self.model = model
        self.cfg = spark_config
        # Khởi tạo SparkContext và StreamingContext với cấu hình đã cho
        conf = SparkConf().setAppName(self.cfg.appName).setMaster(f"{self.cfg.master}[{self.cfg.num_threads}]")
        self.sc = SparkContext(conf=conf)
        self.ssc = StreamingContext(self.sc, self.cfg.batch_interval)
        # Tạo DStream để lắng nghe dữ liệu văn bản từ socket
        self.stream = self.ssc.socketTextStream(self.cfg.hostname, self.cfg.port)

    def start_training(self):
        self.stream.foreachRDD(self._process_batch)
        print("[Receiver] Bắt đầu lắng nghe dữ liệu trên cổng %d ..." % self.cfg.port)
        self.ssc.start()
        self.ssc.awaitTermination()

    def _process_batch(self, time, rdd):
        if rdd.isEmpty():
            return  # Không có dữ liệu trong batch này
        # Parse mỗi dòng JSON (một batch) thành danh sách các mẫu (dict), rồi chuyển thành LabeledPoint
        labeled_rdd = rdd.flatMap(lambda line: json.loads(line)) \
                         .map(lambda record: LabeledPoint(record['y'], Vectors.dense(record['X'])))
        # Huấn luyện mô hình trên RDD vừa tạo
        trained_model = self.model.train(labeled_rdd)
        # In thông tin kết quả huấn luyện
        count = labeled_rdd.count()
        print(f"[Receiver] Batch tại thời điểm {time}: nhận {count} mẫu, đã huấn luyện mô hình {self.model.algorithm}.")
        if trained_model:
            # In ra một vài thông số của mô hình (ví dụ với Logistic Regression là vector trọng số)
            try:
                weights = trained_model.weights  # chỉ áp dụng cho LogisticRegressionModel từ MLlib
                print(f"[Receiver] Số đặc trưng của mô hình: {len(weights)}")
            except Exception:
                # Đối với Decision Tree hoặc Random Forest, có thể in độ sâu hoặc số cây
                if hasattr(trained_model, "numTrees"):
                    print(f"[Receiver] Mô hình Random Forest với {trained_model.numTrees} cây được huấn luyện.")
                elif hasattr(trained_model, "depth"):
                    print(f"[Receiver] Mô hình Decision Tree độ sâu: {trained_model.depth}")
        print("-"*40)
