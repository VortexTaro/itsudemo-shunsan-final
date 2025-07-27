
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
BACKGROUND_IMAGE_PATH = os.path.join(APP_DIR, "assets", "background.jpg")

# --- デザイン設定 ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    bg_image_base64 = get_base64_of_bin_file(BACKGROUND_IMAGE_PATH)
    custom_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap');
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    h1 {{
        color: #FFFFFF;
        text-shadow: 2px 2px 8px #000000;
    }}
    /* その他のスタイル */
    div[data-testid="stChatMessage"] {{
        background-color: rgba(30, 30, 30, 0.85);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}
    div[data-testid="stChatMessage"] p, 
    .stSpinner > div > div {{
        color: #EAEAEA;
        font-family: 'Noto Sans JP', sans-serif;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("背景画像が見つかりませんでした。デフォルトの背景を使用します。")


# --- 初期設定とモデル準備 ---
st.title("いつでもしゅんさん")

try:
    # StreamlitのsecretsからAPIキーを設定
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        raise KeyError("APIキーがsecretsに見つかりません。")
    genai.configure(api_key=api_key)
    
    # 埋め込みモデルと生成モデルを初期化
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro-latest")

except KeyError as e:
    st.error(f"エラー: {e}")
    st.stop()
except Exception as e:
    st.error(f"モデルの初期化中に予期せぬエラーが発生しました: {e}")
    st.stop()


# --- 関数定義 ---
def generate_search_query(prompt, conversation_history):
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
    prompt_template = f"""
あなたは優秀な検索アシスタントです。以下の会話履歴と最後のユーザープロンプトを分析し、ベクトルデータベースから最も関連性の高い情報を引き出すための、簡潔かつ効果的な検索クエリを日本語で生成してください。
【会話履歴】
{history_str}
【最後のユーザープロンプト】
{prompt}
【生成すべき検索クエリ】
"""
    try:
        response = model.generate_content(prompt_template)
        return response.text.strip()
    except Exception:
        return prompt

@st.cache_resource
def load_faiss_index(_embeddings):
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
        
        with st.spinner("情報を検索・生成中..."):
            search_query = generate_search_query(prompt, st.session_state.messages)
            docs_with_scores = db.similarity_search_with_score(search_query, k=5)
            
            context = ""
            for doc, score in docs_with_scores:
                if score < 0.75:
                    context += doc.page_content + "\n\n"
                    sources.append({
                        "file_path": doc.metadata.get("source", "不明"),
                        "score": score,
                    })
            
            system_prompt = f"""
あなたはしゅんさんのAIクローンです。提供された情報を元に、しゅんさんらしく、親しみやすく応答してください。
---
関連情報:
{context if context else "関連情報なし"}
---
ユーザーの質問: {prompt}
"""
            response_stream = model.generate_content(system_prompt, stream=True)
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response, 
        "sources": sources,
        "id": str(uuid.uuid4())
    })
    st.rerun() 