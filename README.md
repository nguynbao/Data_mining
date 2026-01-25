# Hệ Thống Gợi Ý Video Hybrid

Dự án này là một hệ thống gợi ý video hybrid sử dụng nhiều phương pháp data mining và machine learning để phân tích và gợi ý video cho người dùng.

## 📋 Mô tả

Hệ thống sử dụng 3 phương pháp chính:

1. **K-Means Clustering**: Phân nhóm người dùng dựa trên hành vi (thời gian xem, tỷ lệ thích, giờ hoạt động)
2. **Content-based Recommendation**: Gợi ý video dựa trên nội dung sử dụng TF-IDF vectorization
3. **Item-based Collaborative Filtering**: Gợi ý video dựa trên hành vi tương tác của cộng đồng người dùng

## 🚀 Cài đặt

### Yêu cầu

- Python 3.8+
- pip

### Các bước cài đặt

1. Clone hoặc tải dự án về máy

2. Tạo virtual environment (khuyến nghị):
```bash
python -m venv venv
```

3. Kích hoạt virtual environment:
   - Trên macOS/Linux:
   ```bash
   source venv/bin/activate
   ```
   - Trên Windows:
   ```bash
   venv\Scripts\activate
   ```

4. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## 📁 Cấu trúc dự án

```
Data mining/
├── main.py                          # File chính để chạy hệ thống
├── requirements.txt                 # Danh sách các thư viện cần thiết
├── README.md                        # Tài liệu dự án
├── controllers/                     # Các thuật toán gợi ý
│   ├── __init__.py
│   ├── content_based.py            # Content-based recommendation
│   ├── item_based.py               # Item-based collaborative filtering
│   └── kmeans_clustering.py       # K-Means clustering
├── datasets/                        # Dữ liệu và utilities
│   ├── __init__.py
│   ├── data_loader.py              # Hàm tải và xử lý dữ liệu
│   └── tiktok_full_recommendation_dataset.csv  # Dataset
└── venv/                           # Virtual environment (không commit)
```

## 🎯 Sử dụng

### Chạy ứng dụng Web (Streamlit Dashboard)
Đây là giao diện chính thức với UI/UX hiện đại, hỗ trợ Dark/Light mode và Visualization.

```bash
python -m streamlit run app.py
```

### Chạy script CLI (cũ)
```bash
python main.py
```
Hệ thống sẽ:
1. Tải dữ liệu từ file CSV
2. Phân nhóm người dùng bằng K-Means
3. Thực hiện gợi ý content-based
4. Thực hiện gợi ý item-based collaborative filtering

### Sử dụng từng module riêng lẻ

#### 1. Phân nhóm người dùng

```python
from datasets.data_loader import load_data
from controllers.kmeans_clustering import UserManager

videos_df, users_df, interactions_df = load_data()
user_manager = UserManager(users_df)
users_clustered = user_manager.cluster_users(n_clusters=3)
```

#### 2. Content-based Recommendation

```python
from controllers.content_based import ContentRecommender

content_rec = ContentRecommender(videos_df)
recommendations = content_rec.recommend(video_id=1024, top_n=5)
```

#### 3. Item-based Collaborative Filtering

```python
from controllers.item_based import ItemRecommender

item_rec = ItemRecommender(interactions_df)
recommendations = item_rec.recommend(video_id=1024, top_n=5)
```

## 📊 Dữ liệu

Dataset được sử dụng là `tiktok_full_recommendation_dataset.csv` với các cột:

- `userId`: ID người dùng
- `videoId`: ID video
- `title`: Tiêu đề video
- `hashtags`: Hashtags của video
- `watchTime`: Thời gian xem (giây)
- `liked`: Đã thích (0/1)
- `shared`: Đã chia sẻ (0/1)
- `commented`: Đã bình luận (0/1)
- `saved`: Đã lưu (0/1)
- `timestamp`: Thời gian tương tác

## 🔧 Các tính năng

### Error Handling
- Kiểm tra file dữ liệu tồn tại
- Xử lý dữ liệu thiếu hoặc không hợp lệ
- Thông báo lỗi rõ ràng

### Type Hints
- Tất cả các hàm đều có type hints để dễ đọc và maintain

### Documentation
- Docstrings đầy đủ cho tất cả các class và method
- Comments giải thích logic phức tạp

### Code Quality
- Cấu trúc code rõ ràng, dễ đọc
- Tách biệt concerns (separation of concerns)
- Sử dụng best practices của Python

## 📚 Thư viện sử dụng

- **pandas**: Xử lý và phân tích dữ liệu
- **numpy**: Tính toán số học
- **scikit-learn**: Machine learning algorithms (K-Means, TF-IDF, Cosine Similarity)
- **scipy**: Thư viện khoa học tính toán
- **streamlit**: Framework xây dựng Web App
- **plotly/seaborn/matplotlib**: Trực quan hóa dữ liệu (Charts, Heatmaps)

## 🛠️ Cải tiến trong tương lai

- [ ] Thêm hybrid recommendation (kết hợp content-based và item-based)
- [ ] Tối ưu hóa hiệu suất với sparse matrices
- [ ] Thêm evaluation metrics (precision, recall, F1-score)
- [ ] Hỗ trợ real-time recommendation
- [ ] Thêm visualization cho kết quả clustering
- [ ] Tích hợp deep learning models

## 📝 License

Dự án này được tạo cho mục đích học tập và nghiên cứu.

## 👤 Tác giả

Dự án Data Mining - Hệ thống gợi ý video hybrid

## 🙏 Acknowledgments

- Dataset: TikTok Full Recommendation Dataset
- Libraries: scikit-learn, pandas, numpy

