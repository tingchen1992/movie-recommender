import os
import streamlit as st
import pandas as pd
import requests
import ast
from dotenv import load_dotenv

# ======== 載入 API Key ========
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")


# ======== 資料處理工具 ========
def get_top_cast(cast_str):
    try:
        cast = ast.literal_eval(cast_str)
        return [person["name"] for person in cast[:3]]
    except:
        return []


def get_director(crew_str):
    try:
        crew = ast.literal_eval(crew_str)
        return [person["name"] for person in crew if person["job"] == "Director"]
    except:
        return []


def get_genres(genres_str):
    try:
        genres = ast.literal_eval(genres_str)
        return [g["name"] for g in genres]
    except:
        return []


# ======== 快取資料讀取 ========
@st.cache_data
def load_data():
    movies_df = pd.read_csv("tmdb_5000_movies.csv")
    credits_df = pd.read_csv("tmdb_5000_credits.csv")

    movies_df["genres_list"] = movies_df["genres"].apply(get_genres)
    credits_df["top_cast"] = credits_df["cast"].apply(get_top_cast)
    credits_df["director"] = credits_df["crew"].apply(get_director)

    df = pd.merge(
        movies_df,
        credits_df[["movie_id", "top_cast", "director"]],
        left_on="id",
        right_on="movie_id",
    )
    df["tags"] = df.apply(
        lambda row: " ".join(row["genres_list"] + row["director"] + row["top_cast"]),
        axis=1,
    )
    return df


df = load_data()


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


# ======== 推薦系統 ========
@st.cache_data
def recommend_movies(selected_title, top_n=3):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["tags"])
    idx = df[df["title"] == selected_title].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-(top_n + 1) : -1][::-1]
    return df.iloc[similar_indices][["title", "overview"]]


# ======== Streamlit UI ========
st.markdown(
    """
    <div style="height:50px" id="top-anchor"></div>
""",
    unsafe_allow_html=True,
)


st.title("🎬 電影推薦系統 Demo")

# 改成文字輸入框（可自行輸入），自動提示靠接近字串匹配
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

    if st.button("推薦相似電影"):
        with st.spinner("電影推薦中..."):
            recommendations = recommend_movies(movie_title, top_n=3)
            st.subheader("🔍 推薦的相似電影：")
            for idx, row in recommendations.iterrows():
                st.markdown(f"### 🎞️ {row['title']}")
                if pd.isna(row["overview"]) or row["overview"].strip() == "":
                    st.write("無電影簡介")
                else:
                    st.write(row["overview"])
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
