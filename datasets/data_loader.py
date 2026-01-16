"""
Data loader module for TikTok recommendation dataset.
"""
import pandas as pd
from typing import Tuple
import os


def load_data(data_path: str = 'datasets/tiktok_full_recommendation_dataset.csv') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tải và xử lý dữ liệu từ file TikTok full dataset.
    
    Args:
        data_path: Đường dẫn đến file CSV chứa dữ liệu. Mặc định là 'datasets/tiktok_full_recommendation_dataset.csv'
    
    Returns:
        Tuple chứa 3 DataFrames:
            - videos_df: DataFrame chứa thông tin video (video_id, title, hashtags)
            - users_df: DataFrame chứa thông tin người dùng (user_id, total_watch_time, like_ratio, active_hour)
            - interactions_df: DataFrame chứa tương tác người dùng-video (user_id, video_id, rating)
    
    Raises:
        FileNotFoundError: Nếu file dữ liệu không tồn tại
        ValueError: Nếu dữ liệu không có đủ các cột cần thiết
    """
    # Kiểm tra file tồn tại
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {data_path}")
    
    # Tải dữ liệu từ file
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise ValueError(f"Lỗi khi đọc file CSV: {str(e)}")
    
    # Kiểm tra các cột cần thiết
    required_columns = ['videoId', 'title', 'hashtags', 'userId', 'watchTime', 'liked', 'timestamp']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Thiếu các cột cần thiết: {missing_columns}")
    
    # Tạo videos_df: videoId, title, hashtags
    videos_df = df[['videoId', 'title', 'hashtags']].drop_duplicates().rename(columns={'videoId': 'video_id'})
    
    # Tạo users_df: userId, total_watch_time (sum watchTime), like_ratio (mean liked), active_hour (mode từ timestamp hour)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    
    users_agg = df.groupby('userId').agg(
        total_watch_time=('watchTime', 'sum'),
        like_ratio=('liked', 'mean'),  # Tỷ lệ thích trung bình
        active_hour=('hour', lambda x: x.mode().iloc[0] if not x.mode().empty else 12)  # Mode của giờ
    ).reset_index().rename(columns={'userId': 'user_id'})
    users_df = users_agg
    
    # Tạo interactions_df: userId, videoId, rating (tính từ liked, shared, commented, saved)
    # Đảm bảo các cột tồn tại trước khi tính toán
    if 'shared' not in df.columns:
        df['shared'] = 0
    if 'commented' not in df.columns:
        df['commented'] = 0
    if 'saved' not in df.columns:
        df['saved'] = 0
    
    df['rating'] = df['liked'] * 5 + df['shared'] * 3 + df['commented'] * 2 + df['saved'] * 1
    interactions_df = df[['userId', 'videoId', 'rating']].rename(columns={'userId': 'user_id', 'videoId': 'video_id'})
    
    return videos_df, users_df, interactions_df