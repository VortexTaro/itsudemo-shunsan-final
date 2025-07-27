
import streamlit as st
import google.generativeai as genai
import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio
import uuid
import re
import base64

# --- パス設定 ---
# このファイルの場所を基準に、パスを正しく設定
APP_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_DIR = os.path.join(APP_DIR, "knowledge_base")
FAISS_INDEX_PATH = os.path.join(APP_DIR, "data", "faiss_index")
AVATAR_IMAGE_PATH = os.path.join(APP_DIR, "assets", "avatar.png")


# --- デザイン設定 ---
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap');

/* 全体のフォントと背景色 (ヘッダー・フッター・ボトムエリア含む) */
body, .stApp, [data-testid="stHeader"], [data-testid="stFooter"], [data-testid="stBottom"] {
    font-family: 'Noto Sans JP', sans-serif;
    background-color: #1E1E1E !important; /* ダークグレーの背景 */
    color: #EAEAEA; /* 明るいグレーのテキスト */
}

/* アプリのタイトル */
h1 {
    color: #FFFFFF;
    text-shadow: 1px 1px 5px rgba(0,0,0,0.5);
}

/* チャットメッセージのスタイル */
div[data-testid="stChatMessage"] {
    background-color: #2D2D2D; /* やや明るいグレー */
    border-radius: 12px;
    border: 1px solid #444444;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

div[data-testid="stChatMessage"] p {
    color: #EAEAEA;
}

/* チャット入力欄のコンテナ (フッター部分) */
div[data-testid="stChatInput"] {
    background-color: #1E1E1E !important;
    border-top: 1px solid #444444;
}

/* チャット書き込み欄 (シンプル・ダーク調に初期化) */
textarea[data-testid="stChatInputTextArea"] {
    background-color: #2D2D2D;
    color: #EAEAEA;
    border: 1px solid #555555;
    border-radius: 5px;
}

/* プレースホルダーのスタイル */
textarea[data-testid="stChatInputTextArea"]::placeholder {
  color: #888888;
}

/* スピナーのテキスト */
.stSpinner > div > div {
    color: #FFFFFF;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# --- 初期設定とモデル準備 ---
st.title("いつでもしゅんさん")

# モデルの取得（関数化してエラーハンドリングを追加）
def get_model(model_name):
    try:
        return genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"モデルの読み込み中にエラーが発生しました: {e}")
        return None

# APIキーの設定
try:
    # Streamlit SecretsからAPIキーを取得
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except (KeyError, Exception):
    st.error("APIキーが設定されていません。")
    st.stop()

# --- 関数定義 ---
# generate_search_query 関数を完全に削除

@st.cache_resource(show_spinner=False)
def load_faiss_index(_embeddings):
    """
    FAISSインデックスを読み込むか、存在しない場合は作成する。
    """
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, _embeddings, allow_dangerous_deserialization=True)
    
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        st.error(f"ナレッジベースのディレクトリが見つかりません: {KNOWLEDGE_BASE_DIR}")
        st.stop()

    search_path = os.path.join(KNOWLEDGE_BASE_DIR, "**", "*.txt")
    all_file_paths = glob.glob(search_path, recursive=True)
    
    if not all_file_paths:
        st.error(f"ディレクトリ '{KNOWLEDGE_BASE_DIR}' にドキュメント(.txt)が見つかりません。")
        st.stop()
        
    documents = [TextLoader(fp, encoding='utf-8').load()[0] for fp in all_file_paths]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    db = FAISS.from_documents(chunks, _embeddings)
    db.save_local(FAISS_INDEX_PATH)
    return db

db = load_faiss_index(embeddings)

# --- チャット履歴の初期化・表示 ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "僕はしゅんさんのクローンです。しゅんさんが教えてくれた情報を元にあなたの質問に答えちゃうよ！",
        "id": str(uuid.uuid4()),
        "sources": []
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATAR_IMAGE_PATH if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("参照元ファイル"):
                for source in msg["sources"]:
                    st.markdown(f"**{os.path.relpath(source['file_path'], KNOWLEDGE_BASE_DIR)}** (スコア: {source['score']:.2f})")

# --- メインのチャット処理 ---
if prompt := st.chat_input("質問や相談したいことを入力してね"):
    st.session_state.messages.append({"role": "user", "content": prompt, "id": str(uuid.uuid4())})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant", avatar=AVATAR_IMAGE_PATH):
        placeholder = st.empty()
        full_response = ""
        sources = []
        
        with st.spinner("宇宙と接続中だよ！ちょっとまってね..."):
            # 【初期化①】AIによる検索クエリ生成を完全に廃止。ユーザーの入力をそのまま使用する。
            search_query = prompt
            docs_with_scores = db.similarity_search_with_score(search_query, k=3)
            
            relevant_docs_with_scores = [(doc, score) for doc, score in docs_with_scores if score < 0.7]

            context = ""
            if relevant_docs_with_scores:
                sources = [
                    {
                        "id": str(uuid.uuid4()),
                        "file_path": doc.metadata.get("source", "不明"),
                        "score": score,
                        "page_content": doc.page_content,
                    }
                    for doc, score in relevant_docs_with_scores
                ]
                context = "\n\n".join([doc.page_content for doc, score in relevant_docs_with_scores])

            # 【初期化②】システムプロンプトを極限まで単純化。余計な役割は一切与えない。
            system_prompt = f"""
あなたは親切なアシスタントです。以下の「関連情報」を使って、ユーザーの質問に分かりやすく答えてください。
もし関連情報がない場合は、「その件に関する情報は見つかりませんでした」とだけ答えてください。

---
関連情報:
{context if context else "関連情報なし"}
---
ユーザーの質問: {prompt}
"""
            
            try:
                model = get_model("gemini-1.5-pro-latest")
                if model:
                    response_stream = model.generate_content(system_prompt, stream=True)
                    for chunk in response_stream:
                        full_response += chunk.text
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"AI応答生成中にエラーが発生しました: {e}")
                placeholder.markdown("申し訳ありません。エラーが発生しました。")

    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response, 
        "sources": sources,
        "id": str(uuid.uuid4())
    })
    st.rerun() 