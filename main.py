import os
import streamlit as st
import pandas as pd
import requests
import ast
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ======== 載入 API Key ========
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")


# ======== genre 欄位處理 ========
def get_genres(genres_str):
    try:
        genres = ast.literal_eval(genres_str)
        return [g["name"] for g in genres]
    except:
        return []


# ======== 快取資料讀取 ========
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df["genres_list"] = df["genres"].apply(get_genres)
    df["overview"] = df["overview"].fillna("")
    df["tags"] = df.apply(
        lambda row: " ".join(row["genres_list"]) * 3 + " " + row["overview"], axis=1
    )
    return df


df = load_data()


# ======== 快取嵌入模型載入 ========
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# ======== 快取嵌入計算 ========
@st.cache_data
def compute_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)


embeddings = compute_embeddings(df["tags"].tolist())


# ======== 推薦系統主程式 ========
def recommend_movies(selected_title, top_n=3):
    idx = df[df["title"] == selected_title].index[0]
    selected_vec = embeddings[idx].cpu().numpy()
    all_vecs = embeddings.cpu().numpy()
    cosine_sim = cosine_similarity([selected_vec], all_vecs)[0]
    similar_indices = cosine_sim.argsort()[-(top_n + 1) : -1][::-1]
    similar_scores = cosine_sim[similar_indices]
    return df.iloc[similar_indices][["title", "overview"]], similar_scores


# ======== 海報抓取工具 ========
@st.cache_data
def fetch_poster(title):
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    try:
        response = requests.get(search_url, params=params)
        data = response.json()
        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print("抓海報錯誤:", e)
    return None


# ======== Streamlit UI ========
st.markdown('<div style="height:50px" id="top-anchor"></div>', unsafe_allow_html=True)
st.title("🎬 電影推薦系統")

search_query = st.text_input("請輸入電影名稱（可模糊搜尋）")
matched_titles = sorted(
    [title for title in df["title"].unique() if search_query.lower() in title.lower()]
)

if matched_titles:
    movie_title = st.selectbox("請選擇電影（根據你輸入的關鍵字）", matched_titles)
else:
    movie_title = None

if movie_title:
    overview_en = df.loc[df["title"] == movie_title, "overview"].values[0]
    st.write("**英文簡介:**")
    st.write(overview_en)

    poster_url = fetch_poster(movie_title)
    if poster_url:
        st.image(poster_url, caption=movie_title)
    else:
        st.write("找不到電影圖片。")

    if st.button("🎯 推薦相似電影"):
        with st.spinner("電影推薦中..."):
            recommendations, scores = recommend_movies(movie_title, top_n=3)
            st.subheader("🔍 推薦的相似電影")
            for i, (idx, row) in enumerate(recommendations.iterrows()):
                st.markdown(f"### 🎞️ {row['title']}")
                st.write(row["overview"] if pd.notna(row["overview"]) else "無電影簡介")
                rec_poster = fetch_poster(row["title"])
                if rec_poster:
                    st.image(rec_poster, width=200)
                st.markdown("---")

# ======== 回到最上面按鈕 ========
st.markdown(
    """
    <style>
    #back-to-top-btn {
        position: fixed;
        bottom: 40px;
        right: 30px;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 12px 16px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        font-size: 14px;
        z-index: 999;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    </style>

    <a href="#top-anchor" id="back-to-top-btn">⬆ TOP</a>
    """,
    unsafe_allow_html=True,
)
