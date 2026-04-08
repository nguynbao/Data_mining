import time
import pandas as pd
from controllers.kmeans_clustering import UserManager
from controllers.content_based import ContentRecommender
from controllers.item_based import ItemRecommender

def compare_algorithms(videos_df: pd.DataFrame, users_df: pd.DataFrame, interactions_df: pd.DataFrame):
    """
    So sánh thời gian khởi tạo và thực thi của 3 thuật toán:
    1. K-Means Clustering
    2. Content-based Recommendation
    3. Item-based Collaborative Filtering
    """
    print("\n" + "=" * 60)
    print("=== [5/5] SO SÁNH HIỆU NĂNG CÁC THUẬT TOÁN ===")
    print("=" * 60)
    
    results = []

    # --- 1. K-Means Clustering ---
    print("\nĐang đo lường K-Means Clustering...")
    try:
        start_init = time.time()
        user_manager = UserManager(users_df)
        end_init = time.time()
        init_time_kmeans = end_init - start_init
        
        start_exec = time.time()
        user_manager.cluster_users(n_clusters=2)
        end_exec = time.time()
        exec_time_kmeans = end_exec - start_exec
        
        results.append({
            "Thuật toán": "K-Means Clustering",
            "Thời gian khởi tạo (s)": round(init_time_kmeans, 4),
            "Thời gian thực thi (s)": round(exec_time_kmeans, 4),
            "Tổng thời gian (s)": round(init_time_kmeans + exec_time_kmeans, 4)
        })
    except Exception as e:
        print(f"[ERROR] Lỗi K-Means: {str(e)}")
        results.append({
            "Thuật toán": "K-Means Clustering",
            "Thời gian khởi tạo (s)": "Lỗi",
            "Thời gian thực thi (s)": "Lỗi",
            "Tổng thời gian (s)": "Lỗi"
        })

    # --- 2. Content-based ---
    print("\nĐang đo lường Content-based Recommendation...")
    try:
        if videos_df.empty:
            raise ValueError("Không có dữ liệu video")
        
        sample_video_id = videos_df['video_id'].iloc[0]
        
        start_init = time.time()
        content_rec = ContentRecommender(videos_df)
        end_init = time.time()
        init_time_content = end_init - start_init
        
        start_exec = time.time()
        content_rec.recommend(video_id=sample_video_id, top_n=5)
        end_exec = time.time()
        exec_time_content = end_exec - start_exec
        
        results.append({
            "Thuật toán": "Content-based",
            "Thời gian khởi tạo (s)": round(init_time_content, 4),
            "Thời gian thực thi (s)": round(exec_time_content, 4),
            "Tổng thời gian (s)": round(init_time_content + exec_time_content, 4)
        })
    except Exception as e:
        print(f"[ERROR] Lỗi Content-based: {str(e)}")
        results.append({
            "Thuật toán": "Content-based",
            "Thời gian khởi tạo (s)": "Lỗi",
            "Thời gian thực thi (s)": "Lỗi",
            "Tổng thời gian (s)": "Lỗi"
        })

    # --- 3. Item-based ---
    print("\nĐang đo lường Item-based Collaborative Filtering...")
    try:
        if interactions_df.empty or videos_df.empty:
            raise ValueError("Không có dữ liệu interactions hoặc videos")
            
        sample_video_id = videos_df['video_id'].iloc[0]
        
        start_init = time.time()
        item_rec = ItemRecommender(interactions_df)
        end_init = time.time()
        init_time_item = end_init - start_init
        
        start_exec = time.time()
        item_rec.recommend(video_id=sample_video_id, top_n=5)
        end_exec = time.time()
        exec_time_item = end_exec - start_exec
        
        results.append({
            "Thuật toán": "Item-based CF",
            "Thời gian khởi tạo (s)": round(init_time_item, 4),
            "Thời gian thực thi (s)": round(exec_time_item, 4),
            "Tổng thời gian (s)": round(init_time_item + exec_time_item, 4)
        })
    except Exception as e:
        print(f"[ERROR] Lỗi Item-based: {str(e)}")
        results.append({
            "Thuật toán": "Item-based CF",
            "Thời gian khởi tạo (s)": "Lỗi",
            "Thời gian thực thi (s)": "Lỗi",
            "Tổng thời gian (s)": "Lỗi"
        })

    # --- In kết quả dạng bảng ---
    print("\n" + "-" * 80)
    print("KẾT QUẢ SO SÁNH THỜI GIAN THỰC THI (Đơn vị: giây)")
    print("-" * 80)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("-" * 80)
    
    return results_df
