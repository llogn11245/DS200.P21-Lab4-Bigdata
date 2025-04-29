import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class Model:
    """
    Model phân loại chó/mèo.
    Hỗ trợ các thuật toán: Logistic Regression, Decision Tree, Random Forest.
    """
    def __init__(self, algorithm: str = "logistic"):
        if algorithm == "logistic":
            # Logistic Regression với tối đa 200 vòng lặp
            self.model = LogisticRegression(max_iter=200)
        elif algorithm == "decision_tree":
            # Decision Tree mặc định
            self.model = DecisionTreeClassifier()
        elif algorithm == "random_forest":
            # Random Forest với 10 cây
            self.model = RandomForestClassifier(n_estimators=10)
        else:
            raise ValueError(f"Thuật toán không hỗ trợ: {algorithm}")
        self.algorithm = algorithm

    def train(self, training_rdd):
        # Thu thập dữ liệu
        data = training_rdd.collect()
        if not data:
            return None
        # Chuyển thành numpy array
        X = np.array([point.features.toArray() for point in data])
        y = np.array([point.label for point in data])

        # Huấn luyện model
        self.model.fit(X, y)
        return self.model
