"""
Item-based collaborative filtering recommendation system.
"""
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import Union
import numpy as np


class ItemRecommender:
    """
    Hệ thống gợi ý dựa trên Item-based Collaborative Filtering.
    
    Phương pháp này phân tích hành vi người dùng để tìm các video tương tự
    dựa trên việc những người dùng xem video này cũng thường xem video khác.
    """
    
    def __init__(self, interactions_df: pd.DataFrame):
        """
        Khởi tạo ItemRecommender.
        
        Args:
            interactions_df: DataFrame chứa tương tác với các cột: user_id, video_id, rating
        """
        if interactions_df.empty:
            raise ValueError("DataFrame interactions_df không được rỗng")
        
        required_columns = ['user_id', 'video_id', 'rating']
        missing_columns = [col for col in required_columns if col not in interactions_df.columns]
        if missing_columns:
            raise ValueError(f"Thiếu các cột cần thiết: {missing_columns}")
        
        self.interactions_df = interactions_df.copy()
        self.item_sim_df = None
        self._prepare_model()

    def _prepare_model(self) -> None:
        """
        Tạo ma trận User-Item và tính toán ma trận tương đồng giữa các video.
        """
        # Tạo ma trận User-Item
        user_item_matrix = self.interactions_df.pivot_table(
            index='user_id', 
            columns='video_id', 
            values='rating',
            aggfunc='mean'  # Trung bình nếu có nhiều rating cho cùng user-video
        ).fillna(0)
        
        # Kiểm tra nếu ma trận rỗng hoặc chỉ có 1 video
        if user_item_matrix.shape[1] < 2:
            print("[WARNING] Không đủ dữ liệu để tính toán item similarity (cần ít nhất 2 video)")
            self.item_sim_df = pd.DataFrame()
            return
        
        # Tính tương đồng giữa các Video (Transpose matrix)
        item_similarity = cosine_similarity(user_item_matrix.T)
        self.item_sim_df = pd.DataFrame(
            item_similarity, 
            index=user_item_matrix.columns, 
            columns=user_item_matrix.columns
        )

    def recommend(self, video_id: int, top_n: int = 5) -> pd.Series:
        """
        Gợi ý các video tương tự dựa trên video_id.
        
        Args:
            video_id: ID của video cần tìm video tương tự
            top_n: Số lượng video gợi ý (mặc định: 5)
        
        Returns:
            Series chứa video_id và similarity score, được sắp xếp theo độ tương đồng giảm dần
            Trả về Series rỗng nếu không tìm thấy video_id hoặc không có video tương tự
        """
        print(f"\n[INFO] Đang chạy Item-based CF cho Video ID: {video_id}")
        
        if self.item_sim_df is None or self.item_sim_df.empty:
            print("[WARNING] Ma trận similarity chưa được khởi tạo")
            return pd.Series(dtype='float64')
        
        if video_id not in self.item_sim_df.index:
            print(f"[WARNING] Không tìm thấy video với ID: {video_id} trong ma trận similarity")
            return pd.Series(dtype='float64')
            
        # Lấy các video tương đồng nhất
        similar_videos = self.item_sim_df[video_id].sort_values(ascending=False)
        
        # Bỏ qua chính video đó và lọc các video có similarity > 0
        similar_videos = similar_videos.drop(video_id)
        similar_videos = similar_videos[similar_videos > 0]
        
        if similar_videos.empty:
            print(f"[WARNING] Không tìm thấy video tương tự cho video ID: {video_id}")
            return pd.Series(dtype='float64')
        
        return similar_videos.head(top_n)