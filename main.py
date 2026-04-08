"""
Hệ thống gợi ý video hybrid sử dụng nhiều phương pháp:
1. K-Means Clustering: Phân nhóm người dùng
2. Content-based: Gợi ý dựa trên nội dung (TF-IDF)
3. Item-based Collaborative Filtering: Gợi ý dựa trên hành vi người dùng
"""
import sys
from datasets.data_loader import load_data
from controllers.kmeans_clustering import UserManager
from controllers.content_based import ContentRecommender
from controllers.item_based import ItemRecommender
from controllers.comparison import compare_algorithms


def main():
    """
    Hàm main để chạy hệ thống gợi ý video hybrid.
    """
    print("=" * 60)
    print("=== HỆ THỐNG GỢI Ý VIDEO HYBRID ===")
    print("=" * 60)
    
    try:
        # 1. Load dữ liệu
        print("\n[1/5] Đang tải dữ liệu...")
        videos_df, users_df, interactions_df = load_data()
        print(f"✓ Đã load {len(videos_df)} videos, {len(users_df)} users, và {len(interactions_df)} interactions.")
        
        if videos_df.empty or users_df.empty or interactions_df.empty:
            print("[ERROR] Dữ liệu rỗng. Vui lòng kiểm tra file dữ liệu.")
            return
        
    except FileNotFoundError as e:
        print(f"[ERROR] {str(e)}")
        print("Vui lòng đảm bảo file dữ liệu tồn tại trong thư mục datasets/")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Lỗi khi tải dữ liệu: {str(e)}")
        sys.exit(1)

    try:
        # ---------------------------------------------------------
        # CHỨC NĂNG 1: Phân nhóm người dùng (K-Means)
        # ---------------------------------------------------------
        print("\n[2/5] Phân nhóm người dùng (K-Means Clustering)...")
        user_manager = UserManager(users_df)
        users_clustered = user_manager.cluster_users(n_clusters=2)
        print("\n✓ Kết quả phân nhóm (mẫu 10 người dùng đầu tiên):")
        print(users_clustered[['user_id', 'cluster']].head(20))
        
    except Exception as e:
        print(f"[ERROR] Lỗi khi phân nhóm người dùng: {str(e)}")
        users_clustered = None

    try:
        # ---------------------------------------------------------
        # CHỨC NĂNG 2: Gợi ý theo nội dung (TF-IDF)
        # ---------------------------------------------------------
        print("\n[3/5] Gợi ý theo nội dung (Content-based)...")
        if not videos_df.empty:
            sample_video_id = videos_df['video_id'].iloc[1]
            content_rec = ContentRecommender(videos_df)
            results_content = content_rec.recommend(video_id=sample_video_id, top_n=5)
            
            if not results_content.empty:
                print(f"\n✓ Kết quả Content-based (Nếu thích video {sample_video_id} sẽ thích):")
                print(results_content.to_string(index=False))
            else:
                print(f"[WARNING] Không tìm thấy video tương tự cho video {sample_video_id}")
        else:
            print("[WARNING] Không có dữ liệu video để gợi ý")
            
    except Exception as e:
        print(f"[ERROR] Lỗi khi gợi ý content-based: {str(e)}")

    try:
        # ---------------------------------------------------------
        # CHỨC NĂNG 3: Gợi ý theo tương đồng hành vi (Item-based CF)
        # ---------------------------------------------------------
        print("\n[4/5] Gợi ý theo tương đồng hành vi (Item-based CF)...")
        if not interactions_df.empty and not videos_df.empty:
            sample_video_id = videos_df['video_id'].iloc[0]
            item_rec = ItemRecommender(interactions_df)
            results_item = item_rec.recommend(video_id=sample_video_id, top_n=5)
            
            if not results_item.empty:
                print(f"\n✓ Kết quả Item-based (Cộng đồng xem video {sample_video_id} cũng thường xem):")
                print(results_item.to_string())
            else:
                print(f"[WARNING] Không tìm thấy video tương tự cho video {sample_video_id}")
        else:
            print("[WARNING] Không có đủ dữ liệu để gợi ý item-based")
            
    except Exception as e:
        print(f"[ERROR] Lỗi khi gợi ý item-based: {str(e)}")
        
    try:
        # ---------------------------------------------------------
        # CHỨC NĂNG 4: So sánh 3 thuật toán
        # ---------------------------------------------------------
        compare_algorithms(videos_df, users_df, interactions_df)
            
    except Exception as e:
        print(f"[ERROR] Lỗi khi so sánh thuật toán: {str(e)}")
    
    print("\n" + "=" * 60)
    print("=== HOÀN THÀNH ===")
    print("=" * 60)


if __name__ == "__main__":
    main()