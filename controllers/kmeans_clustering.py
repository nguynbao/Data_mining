"""
K-Means clustering for user segmentation.
"""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Optional
import numpy as np


class UserManager:
    """
    Quản lý và phân nhóm người dùng sử dụng K-Means clustering.
    
    Phương pháp này phân nhóm người dùng dựa trên các đặc trưng:
    - total_watch_time: Tổng thời gian xem
    - like_ratio: Tỷ lệ thích trung bình
    - active_hour: Giờ hoạt động chính
    """
    
    def __init__(self, users_df: pd.DataFrame):
        """
        Khởi tạo UserManager.
        
        Args:
            users_df: DataFrame chứa thông tin người dùng với các cột: user_id, total_watch_time, like_ratio, active_hour
        """
        if users_df.empty:
            raise ValueError("DataFrame users_df không được rỗng")
        
        required_columns = ['total_watch_time', 'like_ratio', 'active_hour']
        missing_columns = [col for col in required_columns if col not in users_df.columns]
        if missing_columns:
            raise ValueError(f"Thiếu các cột cần thiết: {missing_columns}")
        
        self.users_df = users_df.copy()
        self.scaler = StandardScaler()
        self.kmeans_model = None

    def cluster_users(self, n_clusters: int = 2, random_state: int = 100) -> pd.DataFrame:
        """
        Phân nhóm người dùng sử dụng K-Means clustering.
        
        Args:
            n_clusters: Số lượng nhóm cần phân (mặc định: 2)
            random_state: Random state để đảm bảo kết quả có thể tái tạo (mặc định: 42)
        
        Returns:
            DataFrame với cột 'cluster' được thêm vào, chứa nhóm của mỗi người dùng
        """
        print(f"\n[INFO] Đang chạy K-Means Clustering với {n_clusters} nhóm...")
        
        if n_clusters < 2:
            raise ValueError("Số lượng nhóm phải >= 2")
        
        if len(self.users_df) < n_clusters:
            print(f"[WARNING] Số lượng người dùng ({len(self.users_df)}) nhỏ hơn số nhóm ({n_clusters}). Sử dụng số nhóm = {len(self.users_df)}")
            n_clusters = len(self.users_df)
        
        # Lấy đặc trưng & Chuẩn hóa
        features = self.users_df[['total_watch_time', 'like_ratio', 'active_hour']].copy()
        
        # Xử lý giá trị NaN
        if features.isna().any().any():
            print("[WARNING] Phát hiện giá trị NaN, thay thế bằng giá trị trung bình")
            features = features.fillna(features.mean())
        
        features_scaled = self.scaler.fit_transform(features)
        
        # Phân cụm
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        clusters = self.kmeans_model.fit_predict(features_scaled)
        
        # Thêm cột cluster vào DataFrame (tạo copy để tránh warning)
        result_df = self.users_df.copy()
        result_df['cluster'] = clusters
        
        print(f"[INFO] Hoàn thành phân nhóm. Phân bố nhóm:")
        print(result_df['cluster'].value_counts().sort_index())
        
        return result_df
    
    def predict_cluster(self, user_features: pd.DataFrame) -> np.ndarray:
        """
        Dự đoán nhóm cho người dùng mới dựa trên mô hình đã huấn luyện.
        
        Args:
            user_features: DataFrame chứa đặc trưng của người dùng với các cột: total_watch_time, like_ratio, active_hour
        
        Returns:
            Array chứa nhóm dự đoán cho mỗi người dùng
        """
        if self.kmeans_model is None:
            raise ValueError("Mô hình chưa được huấn luyện. Gọi cluster_users() trước.")
        
        features = user_features[['total_watch_time', 'like_ratio', 'active_hour']].copy()
        features = features.fillna(features.mean())
        features_scaled = self.scaler.transform(features)
        
        return self.kmeans_model.predict(features_scaled)