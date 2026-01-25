
import streamlit as st
import pandas as pd
import numpy as np
import base64
from datasets.data_loader import load_data
from controllers.kmeans_clustering import UserManager
from controllers.content_based import ContentRecommender
from controllers.item_based import ItemRecommender
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tiktok Hybrid Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
def local_css():
    st.markdown("""
    <style>
        /* IMPORT FONTS & ICONS */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        @import url('https://unpkg.com/@phosphor-icons/web@2.0.3/src/regular/style.css');
        @import url('https://unpkg.com/@phosphor-icons/web@2.0.3/src/fill/style.css');
        
        /* GLOBAL RESET & BASICS */
        html, body, [class*="css"]  {
            font-family: 'Outfit', sans-serif;
        }
        
        /* BACKGROUND GRADIENT - REMOVED for Native Dark/Light Mode */

        /* CUSTOM SCROLLBAR -- Optional, keeping for nice feel but can remove if conflicts */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: #888; 
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #555; 
        }

        /* HEADERS */
        /* Let Streamlit handle colors for dark/light mode */
        
        /* CARDS / GLASSMORPHISM */
        
        /* SIDEBAR - Native is fine */
        
        /* DATAFRAME */
        .stDataFrame {
            /* clear custom bg to let theme handle it */
        }
        
        /* ICONS styling */
        .ph, .ph-fill {
            vertical-align: middle;
            font-size: 1.5rem;
        }
        
    </style>
    """, unsafe_allow_html=True)

local_css()

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data
def get_data():
    """Load and cache data to improve performance."""
    try:
        videos, users, interactions = load_data()
        return videos, users, interactions
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return None, None, None

def create_card(title, value, description="", icon_class="ph ph-chart-bar"):
    """Helper to create a nice HTML metric card."""
    st.markdown(f"""
    <div style="padding: 10px;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 5px;">
            <span style="font-size: 1.1rem; font-weight: 600;">{title}</span>
            <i class="{icon_class}" style="font-size: 1.5rem; color: #00d2ff;"></i>
        </div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #00d2ff;">{value}</div>
        <div style="font-size: 0.8rem; opacity: 0.7;">{description}</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# --- HEADER ---
st.markdown("<h1><i class='ph-fill ph-music-notes-simple' style='color: #00d2ff; margin-right: 10px;'></i>Tiktok Hybrid Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; opacity: 0.8;'>Hệ thống gợi ý video thông minh sử dụng K-Means, Content-based & Item-based Filtering</p>", unsafe_allow_html=True)
st.markdown("---")

# --- LOAD DATA ---
with st.spinner("Đang tải dữ liệu hệ thống..."):
    videos_df, users_df, interactions_df = get_data()

if videos_df is None:
    st.stop()

# --- SIDEBAR ---
st.sidebar.markdown("## <i class='ph ph-compass'></i> ĐIỀU HƯỚNG", unsafe_allow_html=True)
page = st.sidebar.radio("Chọn chức năng", ["Tổng quan Dashboard", "Phân cụm User", "Content-Based", "Item-Based"])

st.sidebar.markdown("---")
st.sidebar.info(f"**Dữ liệu hiện tại:**\n\n- Videos: {len(videos_df)}\n\n- Users: {len(users_df)}\n\n- Tương tác: {len(interactions_df)}")

# -----------------------------------------------------------------------------
# PAGE 1: TỔNG QUAN
# -----------------------------------------------------------------------------
if page == "Tổng quan Dashboard":
    st.markdown("### <i class='ph ph-kanban'></i> Tổng quan hệ thống", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        create_card("Tổng Video", f"{len(videos_df):,}", "Video có sẵn trong kho", "ph ph-film-strip")
    with col2:
        create_card("Active Users", f"{len(users_df):,}", "Người dùng đã tương tác", "ph ph-users")
    with col3:
        create_card("Total Interactions", f"{len(interactions_df):,}", "Lượt thích, share, comment...", "ph ph-heart")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- CHARTS ---
    st.markdown("#### 📊 Phân bố dữ liệu")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Watch Time Distribution
        fig_hist = px.histogram(users_df, x="total_watch_time", nbins=20, 
                                title="Phân bố Tổng thời gian xem (User)",
                                color_discrete_sequence=['#00d2ff'])
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_chart2:
        # Interactions Type Pie Chart (Mock or Real if available, here using Category distribution if available, else Mock interaction types)
        # Using Category from videos if available
        if 'category' in videos_df.columns:
             cat_counts = videos_df['category'].value_counts().reset_index()
             cat_counts.columns = ['category', 'count']
             fig_pie = px.pie(cat_counts, values='count', names='category', 
                              title="Tỷ lệ Video theo Danh mục",
                              color_discrete_sequence=px.colors.sequential.RdBu)
             fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
             st.plotly_chart(fig_pie, use_container_width=True)
        else:
            # Fallback to Top Hashtags bar chart
            all_hashtags = []
            for tags in videos_df['hashtags'].dropna():
                all_hashtags.extend(tags.replace("'", "").replace("[", "").replace("]", "").split(","))
            from collections import Counter
            tag_counts = Counter([t.strip() for t in all_hashtags if t.strip()]).most_common(10)
            tag_df = pd.DataFrame(tag_counts, columns=['Hashtag', 'Count'])
            
            fig_bar = px.bar(tag_df, x='Count', y='Hashtag', orientation='h', title="Top 10 Hashtags",
                             color='Count', color_continuous_scale='Bluered')
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig_bar, use_container_width=True)
            
    st.markdown("---")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("#### Preview Dữ liệu Video")
        st.dataframe(videos_df.head(10), use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("#### Top Hashtags")
        # Simple extraction of hashtags for visualization
        all_hashtags = []
        for tags in videos_df['hashtags'].dropna():
            all_hashtags.extend(tags.replace("'", "").replace("[", "").replace("]", "").split(","))
        
        # Clean and count (simple mock logic for display)
        from collections import Counter
        tag_counts = Counter([t.strip() for t in all_hashtags if t.strip()])
        top_tags = tag_counts.most_common(5)
        
        for tag, count in top_tags:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; border-bottom: 1px solid rgba(128,128,128,0.2); padding-bottom: 4px;">
                <span style="color: #00d2ff;">#{tag}</span>
                <span>{count}</span>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 2: PHÂN CỤM USER (K-MEANS)
# -----------------------------------------------------------------------------
elif page == "Phân cụm User":
    st.markdown("### <i class='ph ph-users-three'></i> Phân nhóm người dùng (K-Means)", unsafe_allow_html=True)
    
    col_control, col_display = st.columns([1, 2])
    
    with col_control:
        st.subheader("Cấu hình")
        n_clusters = st.slider("Số lượng cụm (k)", min_value=2, max_value=10, value=2)
        
        if st.button("Chạy phân cụm"):
            with st.spinner("Đang phân tích..."):
                try:
                    user_manager = UserManager(users_df)
                    users_clustered = user_manager.cluster_users(n_clusters=n_clusters)
                    st.session_state['users_clustered'] = users_clustered
                    st.success("Đã phân cụm thành công!")
                except Exception as e:
                    st.error(f"Lỗi: {e}")

    with col_display:
        if 'users_clustered' in st.session_state:
            st.markdown("#### Kết quả phân cụm")
            st.info("""
            **Giải thích kết quả:**
            Hệ thống đã gom người dùng thành các nhóm có hành vi tương đồng (dựa trên thời gian xem, tỷ lệ like...).
            Bạn có thể thấy sự phân bố số lượng người dùng trong biểu đồ bên dưới.
            """)
            clustered_df = st.session_state['users_clustered']
            
            # --- CLUSTER PROFILING ---
            # Calculate average stats for each cluster
            cluster_profile = clustered_df.groupby('cluster')[['total_watch_time', 'like_ratio', 'active_hour']].mean().reset_index()
            cluster_profile['user_count'] = clustered_df.groupby('cluster').size().reset_index(name='user_count')['user_count']
            
            st.markdown("### 📊 Phân tích đặc điểm từng nhóm")
            col_prof1, col_prof2 = st.columns([1, 2])
            
            with col_prof1:
                st.dataframe(cluster_profile.style.highlight_max(axis=0, color='#00d2ff20'), use_container_width=True)
                st.info("""
                **Cách đọc bảng phân tích:**
                - **total_watch_time cao**: Nhóm "nghiện" xem (High Engagement).
                - **like_ratio cao**: Nhóm hay tương tác/thích.
                """)
            
            with col_prof2:
                 # VISUALIZATION: SCATTER PLOT
                 fig_scatter = px.scatter(
                     clustered_df, 
                     x='total_watch_time', 
                     y='like_ratio', 
                     color='cluster',
                     size='active_hour',
                     hover_data=['user_id'],
                     title="Biểu đồ phân bố cụm (Watch Time vs Like Ratio)",
                     color_continuous_scale=px.colors.qualitative.Bold
                 )
                 fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.05)", font_color="white")
                 st.plotly_chart(fig_scatter, use_container_width=True)

            # Show stats per cluster chart count
            st.bar_chart(clustered_df['cluster'].value_counts())
            
            st.divider()
            st.write("Chi tiết dữ liệu đã gán nhãn:")
            # Format float columns
            display_df = clustered_df[['user_id', 'total_watch_time', 'like_ratio', 'active_hour', 'cluster']].head(50)
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("Vui lòng chọn số cụm và nhấn 'Chạy phân cụm'")

# -----------------------------------------------------------------------------
# PAGE 3: CONTENT-BASED
# -----------------------------------------------------------------------------
elif page == "Content-Based":
    st.markdown("### <i class='ph ph-article'></i> Gợi ý theo nội dung", unsafe_allow_html=True)
    
    if videos_df.empty:
        st.warning("Không có dữ liệu video.")
        st.stop()

    col_sel, col_res = st.columns([1, 2])
    
    with col_sel:
        st.subheader("Chọn Video nguồn")
        
        # Select box with video titles
        video_options = videos_df['video_id'].astype(str).tolist()
        # Create a mapping for better display
        video_map = pd.Series(videos_df['title'].values, index=videos_df['video_id'].astype(str)).to_dict()
        
        selected_video_id = st.selectbox(
            "Chọn ID Video", 
            video_options,
            format_func=lambda x: f"{x} - {video_map.get(x, '')[:30]}..."
        )
        
        top_n = st.slider("Số lượng gợi ý", 1, 10, 5)
        
        run_content = st.button("Gợi ý ngay (Content)")
        
        # Show details of selected video
        if selected_video_id:
            # FIX: Robust filtering
            # Convert both column and value to string for comparison to avoid type mismatch
            mask = videos_df['video_id'].astype(str) == str(selected_video_id)
            sel_rows = videos_df[mask]
            
            if not sel_rows.empty:
                # Use iloc[0] safely because we checked not empty
                sel_row = sel_rows.iloc[0]
                st.markdown(f"""
                <div class="glass-card" style="margin-top: 10px; font-size: 0.9rem;">
                    <b>Video đang chọn:</b><br>
                    Title: <i style="color: #00d2ff;">{sel_row['title']}</i><br>
                    Tags: {sel_row['hashtags']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Không tìm thấy thông tin video.")

    with col_res:
        if run_content:
            st.markdown(f"#### Kết quả gợi ý cho: `{selected_video_id}`")
            st.info(f"""
            **Tại sao lại gợi ý những video này?**
            Hệ thống tìm thấy các video có **Hashtags và Tiêu đề** tương tự với video `{selected_video_id}`.
            Sử dụng công nghệ đo lường tương đồng văn bản (TF-IDF & Cosine Similarity).
            """)
            
            try:
                # Prepare input ID: Try to convert back to original type if needed, 
                # but controllers usually handle it if passed correctly. 
                # Let's inspect the type of video_id in dataframe.
                first_val = videos_df['video_id'].iloc[0]
                
                input_id = selected_video_id
                if isinstance(first_val, (int, np.integer)):
                     try:
                        input_id = int(selected_video_id)
                     except:
                        pass # keep as string if conversion fails
                
                content_rec = ContentRecommender(videos_df)
                results = content_rec.recommend(video_id=input_id, top_n=top_n)
                
                if not results.empty:
                    for i, (idx, row) in enumerate(results.iterrows()):
                        st.markdown(f"""
                        <div style="background: rgba(128,128,128,0.1); padding: 10px; border-radius: 8px; margin-bottom: 8px; border-left: 2px solid #00d2ff;">
                            <strong>#{i+1} - ID: {row['video_id']}</strong><br>
                            <span style="font-size: 1.1em;">{row['title']}</span><br>
                            <span style="font-size: 0.8em; opacity: 0.7;">{row['hashtags']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Không tìm thấy video tương tự.")
            except Exception as e:
                st.error(f"Lỗi: {e}")

# -----------------------------------------------------------------------------
# PAGE 4: ITEM-BASED
# -----------------------------------------------------------------------------
elif page == "Item-Based":
    st.markdown("### <i class='ph ph-handshake'></i> Gợi ý Item-based (Collaborative Filtering)", unsafe_allow_html=True)
    
    if interactions_df.empty:
        st.warning("Không có dữ liệu tương tác.")
        st.stop()

    col_sel_i, col_res_i = st.columns([1, 2])
    
    with col_sel_i:
        st.subheader("Chọn Video nguồn")
        
        video_options = videos_df['video_id'].astype(str).tolist()
        # Reuse mapping
        video_map = pd.Series(videos_df['title'].values, index=videos_df['video_id'].astype(str)).to_dict()
        
        selected_video_id_ib = st.selectbox(
            "Chọn ID Video ", 
            video_options,
            format_func=lambda x: f"{x} - {video_map.get(x, '')[:30]}...",
            key="ib_select"
        )
        
        top_n_ib = st.slider("Số lượng gợi ý", 1, 10, 5, key="ib_slider")
        
        run_item = st.button("Gợi ý ngay (Item-based)")

    with col_res_i:
        if run_item:
            st.markdown(f"#### Kết quả gợi ý cho: `{selected_video_id_ib}`")
            st.info(f"""
            **Tại sao lại gợi ý những video này?**
            Dựa trên dữ liệu lịch sử: **Cộng đồng người dùng** đã xem/thích video `{selected_video_id_ib}` cũng thường xuyên xem/thích các video dưới đây.
            (Độ tương quan càng cao nghĩa là xu hướng xem cùng nhau càng lớn).
            """)
            
            try:
                # Type conversion logic
                first_val = videos_df['video_id'].iloc[0]
                input_id = selected_video_id_ib
                if isinstance(first_val, (int, np.integer)):
                     try:
                        input_id = int(selected_video_id_ib)
                     except:
                        pass
                
                item_rec = ItemRecommender(interactions_df)
                results = item_rec.recommend(video_id=input_id, top_n=top_n_ib)
                
                if not results.empty:
                    # Convert Series to DataFrame for merging
                    results_df = results.reset_index()
                    results_df.columns = ['video_id', 'correlation']
                    
                    # Enrich with titles
                    results_enriched = results_df.merge(videos_df[['video_id', 'title']], on='video_id', how='left')
                    
                    col_list, col_heat = st.columns([1, 1])
                    
                    with col_list:
                        st.write("##### 📋 Danh sách Top Video")
                        for i, (idx, row) in enumerate(results_enriched.iterrows()):
                            st.markdown(f"""
                            <div style="background: rgba(128,128,128,0.1); padding: 10px; border-radius: 8px; margin-bottom: 8px; border-left: 2px solid #e100ff;">
                                <strong>#{i+1} - ID: {row['video_id']}</strong><br>
                                <span style="font-size: 1.1em;">{row['title'] if pd.notna(row['title']) else 'Unknown Title'}</span><br>
                                <div style="text-align: right; color: #e100ff; font-size: 0.8em;">Độ tương quan: {row['correlation']:.4f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col_heat:
                        st.write("##### 🔥 Bản đồ nhiệt tương quan (Heatmap)")
                        # Visualization: Heatmap of recommendation vs Input
                        try:
                            # We create a small matrix of input ID + recommended IDs
                            rec_ids = results_enriched['video_id'].tolist()
                            all_ids = [input_id] + rec_ids
                            
                            # Filter interaction df for these videos
                            subset_interactions = interactions_df[interactions_df['video_id'].isin(all_ids)]
                            
                            if not subset_interactions.empty:
                                pivot_subset = subset_interactions.pivot_table(index='user_id', columns='video_id', values='rating').fillna(0)
                                if pivot_subset.shape[1] > 1:
                                    corr_subset = pivot_subset.corr()
                                    
                                    fig_heat, ax = plt.subplots(figsize=(6, 5))
                                    # Dark background for plot
                                    fig_heat.patch.set_facecolor('#0e1117')
                                    ax.set_facecolor('#0e1117')
                                    
                                    sns.heatmap(corr_subset, annot=True, cmap='coolwarm', fmt=".2f", ax=ax,
                                                cbar_kws={'label': 'Correlation'})
                                    
                                    # Adjust text colors
                                    cbar = ax.collections[0].colorbar
                                    cbar.ax.yaxis.set_tick_params(color='white')
                                    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
                                    ax.tick_params(colors='white', which='both')
                                    ax.xaxis.label.set_color('white')
                                    ax.yaxis.label.set_color('white')
                                    
                                    st.pyplot(fig_heat)
                                else:
                                    st.info("Không đủ dữ liệu chung để vẽ Heatmap.")
                            else:
                                st.info("Không có dữ liệu tương tác chung.")
                        except Exception as ex:
                            st.error(f"Lỗi vẽ biểu đồ: {ex}")
                else:
                    st.warning("Không tìm thấy dữ liệu tương quan hoặc chưa đủ tương tác.")
            except Exception as e:
                st.error(f"Lỗi: {e}")
