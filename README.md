# Hệ Thống Gợi Ý Video Hybrid

Dự án này là một hệ thống gợi ý video hybrid sử dụng nhiều phương pháp data mining và machine learning để phân tích và gợi ý video cho người dùng.

## 📋 Mô tả

Hệ thống sử dụng 3 phương pháp chính để phân tích dữ liệu và gợi ý video:

### 1. K-Means Clustering

Thuật toán này dùng để **phân nhóm người dùng theo hành vi**. Trong dự án, mỗi người dùng được biểu diễn bởi 3 đặc trưng:

- `total_watch_time`: Tổng thời gian xem video
- `like_ratio`: Tỷ lệ thích trung bình
- `active_hour`: Khung giờ hoạt động nhiều nhất

Các đặc trưng này được **chuẩn hóa bằng `StandardScaler`** trước khi đưa vào mô hình `KMeans`. Sau đó, hệ thống gán mỗi người dùng vào một cụm (`cluster`) để nhận diện các nhóm người dùng có hành vi tương đồng. Kết quả này phù hợp cho bài toán phân khúc người dùng và làm nền tảng cho các chiến lược cá nhân hóa.

### 2. Content-based Recommendation

Thuật toán này dùng để **gợi ý video tương tự về mặt nội dung**. Dự án khai thác cột `hashtags` của từng video, sau đó:

- Tiền xử lý dữ liệu thiếu bằng cách thay `NaN` thành chuỗi rỗng
- Biểu diễn nội dung bằng **TF-IDF Vectorization**
- Tính mức độ giống nhau giữa các video bằng **Cosine Similarity**

Khi người dùng chọn một `video_id`, hệ thống sẽ tìm các video có vector nội dung gần nhất với video đó. Cách tiếp cận này phù hợp khi muốn gợi ý các video có chủ đề, từ khóa hoặc hashtag tương tự.

### 3. Item-based Collaborative Filtering

Thuật toán này dùng để **gợi ý video dựa trên hành vi tương tác của cộng đồng người dùng**. Trước tiên, dữ liệu tương tác được chuyển thành ma trận `user-item`, trong đó:

- Hàng là người dùng (`user_id`)
- Cột là video (`video_id`)
- Giá trị là `rating`

Trong dự án, `rating` được xây dựng từ các hành vi:

- `liked * 5`
- `shared * 3`
- `commented * 2`
- `saved * 1`

Sau khi có ma trận tương tác, hệ thống tính **Cosine Similarity giữa các video** để tìm ra những video thường được nhóm người dùng giống nhau quan tâm. Khi nhập một `video_id`, hệ thống trả về các video có mức độ tương đồng cao nhất theo hành vi thực tế của người dùng.

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
