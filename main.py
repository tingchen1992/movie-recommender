import os
import streamlit as st
import pandas as pd
import requests
import ast
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

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

# ======== API 搜尋新片 ========
def get_movie_from_api(title):
    """從 TMDB API 抓取電影資訊（如果 CSV 沒有的話）"""
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title, "language": "en-US"}
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("results"):
        movie = data["results"][0]
        overview = movie.get("overview", "")
        movie_id = movie["id"]

        detail_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        detail_params = {"api_key": TMDB_API_KEY, "language": "en-US"}
        detail_resp = requests.get(detail_url, params=detail_params).json()
        genres = []
        if "genres" in detail_resp:
            genres = [g["name"] for g in detail_resp["genres"]]

        return {
            "title": movie["title"],
            "overview": overview,
            "genres_list": genres,
            "poster_path": movie.get("poster_path"),
        }
    return None

# ======== 動態推薦系統（排除自己選的電影） ========
def recommend_movies_dynamic(title, top_n=3):
    if title in df["title"].values:
        idx = df[df["title"] == title].index[0]
        selected_vec = embeddings[idx].cpu().numpy()
        overview = df.loc[idx, "overview"]
        poster_url = fetch_poster(title)
    else:
        movie_info = get_movie_from_api(title)
        if not movie_info:
            return None, None, None, "找不到電影"

        tags = " ".join(movie_info["genres_list"]) * 3 + " " + movie_info["overview"]
        selected_vec = model.encode([tags])[0]
        overview = movie_info["overview"]
        poster_url = (
            f"https://image.tmdb.org/t/p/w500{movie_info['poster_path']}"
            if movie_info["poster_path"]
            else None
        )

    all_vecs = embeddings.cpu().numpy()
    cosine_sim = cosine_similarity([selected_vec], all_vecs)[0]
    similar_indices = cosine_sim.argsort()[::-1]
    similar_indices = [i for i in similar_indices if df.iloc[i]["title"] != title]
    similar_indices = similar_indices[:top_n]
    similar_scores = cosine_sim[similar_indices]

    return (
        df.iloc[similar_indices][["title", "overview"]],
        similar_scores,
        (overview, poster_url),
        None,
    )

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

# ======== 繁體中文翻譯函式 ========
@st.cache_data
def translate_to_zh_tw(text):
    if not text:
        return ""
    try:
        result = GoogleTranslator(source='auto', target='zh-TW').translate(text)
        return result
    except Exception as e:
        print("翻譯錯誤:", e)
        return text

# ======== Streamlit UI ========
st.markdown('<div style="height:50px" id="top-anchor"></div>', unsafe_allow_html=True)
st.title("🎬 電影推薦系統（支援最新電影）")

search_query = st.text_input("請輸入電影名稱（支援舊片與最新電影）")

matched_titles = sorted(
    [title for title in df["title"].unique() if search_query.lower() in title.lower()]
)

if matched_titles:
    movie_title = st.selectbox("請選擇電影", matched_titles)
else:
    movie_title = search_query if search_query else None
    if search_query and not matched_titles:
        st.info(f"⚡ 嘗試從 TMDB API 搜尋 **{search_query}** ...")

if movie_title:
    with st.spinner("抓取電影資訊中..."):
        recommendations, scores, movie_info, error_msg = recommend_movies_dynamic(
            movie_title, top_n=3
        )

    if error_msg:
        st.error(error_msg)
    else:
        overview_en, poster_url = movie_info

        st.write("**英文簡介:**")
        st.write(overview_en)

        overview_zh = translate_to_zh_tw(overview_en)
        st.write("**繁體中文簡介:**")
        st.write(overview_zh)

        if poster_url:
            st.image(poster_url, caption=movie_title)
        else:
            st.write("找不到電影圖片。")

        if st.button("🎯 推薦相似電影"):
            with st.spinner("電影推薦中..."):
                st.subheader("🔍 推薦的相似電影")
                for i, (idx, row) in enumerate(recommendations.iterrows()):
                    st.markdown(f"### 🎞️ {row['title']}")
                    overview_en = row["overview"] if pd.notna(row["overview"]) else "無電影簡介"
                    st.write("**英文簡介:**")
                    st.write(overview_en)
                    overview_zh = translate_to_zh_tw(overview_en)
                    st.write("**繁體中文簡介:**")
                    st.write(overview_zh)
                    rec_poster = fetch_poster(row["title"])
                    if rec_poster:
                        st.image(rec_poster, width=200)
                    st.markdown("---")

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
