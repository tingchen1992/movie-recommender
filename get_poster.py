import os
import requests
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")


def get_movie_poster_url(title):
    """
    根據電影名稱從 TMDB 搜尋海報 URL。
    找不到時回傳 None。
    """
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    try:
        res = requests.get(search_url, params=params)
        data = res.json()
        if data["results"]:
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return None
    except Exception as e:
        print("海報取得錯誤：", e)
        return None
