"""
Content-based recommendation system using TF-IDF and cosine similarity.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import Optional
import numpy as np


class ContentRecommender:
    """
    Hệ thống gợi ý dựa trên nội dung sử dụng TF-IDF vectorization và cosine similarity.
    
    Phương pháp này phân tích hashtags của video để tìm các video tương tự về mặt nội dung.
    """
    
    def __init__(self, videos_df: pd.DataFrame):
        """
        Khởi tạo ContentRecommender.
        
        Args:
            videos_df: DataFrame chứa thông tin video với các cột: video_id, title, hashtags
        """
        if videos_df.empty:
            raise ValueError("DataFrame videos_df không được rỗng")
        
        if 'hashtags' not in videos_df.columns:
            raise ValueError("DataFrame phải có cột 'hashtags'")
        
        self.videos_df = videos_df.copy()
        self.tfidf_matrix = None
        self.cosine_sim = None
        self._prepare_model()

    def _prepare_model(self) -> None:
        """
        Huấn luyện mô hình TF-IDF và tính toán ma trận cosine similarity.
        """
        # Xử lý giá trị NaN trong hashtags
        self.videos_df['hashtags'] = self.videos_df['hashtags'].fillna('')
        
        # Huấn luyện mô hình TF-IDF
        tfidf = TfidfVectorizer(stop_words='english', min_df=1)
        self.tfidf_matrix = tfidf.fit_transform(self.videos_df['hashtags'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def recommend(self, video_id: int, top_n: int = 5) -> pd.DataFrame:
        """
        Gợi ý các video tương tự dựa trên video_id.
        
        Args:
            video_id: ID của video cần tìm video tương tự
            top_n: Số lượng video gợi ý (mặc định: 5)
        
        Returns:
            DataFrame chứa các video được gợi ý với các cột: video_id, title, hashtags
            Trả về DataFrame rỗng nếu không tìm thấy video_id hoặc không có video tương tự
        """
        print(f"\n[INFO] Đang chạy Content-based cho Video ID: {video_id}")
        
        try:
            # Tìm index của video trong DataFrame
            video_mask = self.videos_df['video_id'] == video_id
            if not video_mask.any():
                print(f"[WARNING] Không tìm thấy video với ID: {video_id}")
                return pd.DataFrame(columns=['video_id', 'title', 'hashtags'])
            
            idx = self.videos_df[video_mask].index[0]
        except (IndexError, KeyError) as e:
            print(f"[ERROR] Lỗi khi tìm video: {str(e)}")
            return pd.DataFrame(columns=['video_id', 'title', 'hashtags'])

        # Lấy điểm tương đồng
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Bỏ qua chính video đó và lấy top_n video
        sim_scores = sim_scores[1:top_n+1]
        
        # Lọc các video có điểm similarity > 0
        sim_scores = [(i, score) for i, score in sim_scores if score > 0]
        
        if not sim_scores:
            print(f"[WARNING] Không tìm thấy video tương tự cho video ID: {video_id}")
            return pd.DataFrame(columns=['video_id', 'title', 'hashtags'])
        
        video_indices = [i[0] for i in sim_scores]
        return self.videos_df.iloc[video_indices][['video_id', 'title', 'hashtags']].copy()