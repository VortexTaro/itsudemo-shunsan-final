
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

/* 全体のフォントと背景色 (ヘッダー・フッター含む) */
body, .stApp, [data-testid="stHeader"], [data-testid="stFooter"] {
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

try:
    # StreamlitのsecretsからAPIキーを設定
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        raise KeyError("APIキーがsecretsに見つかりません。")
    genai.configure(api_key=api_key)
    
    # 埋め込みモデルと生成モデルを初期化
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-pro")

except KeyError as e:
    st.error(f"エラー: {e}")
    st.stop()
except Exception as e:
    st.error(f"モデルの初期化中に予期せぬエラーが発生しました: {e}")
    st.stop()


# --- 関数定義 ---
def generate_search_query(prompt, history):
    """
    ユーザーのプロンプトと会話履歴から、FAISS検索に最適なキーワードを生成する。
    """
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    
    # 履歴を整形
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    
    prompt_template = f"""
あなたは、ユーザーの質問を分析し、情報検索クエリを生成するプロフェッショナルです。以下の指示に厳格に従ってください。

**指示:**
1.  ユーザーの質問から、最も中心的で具体的な「名詞」を最大3つまで抽出します。
2.  感情的な表現、挨拶、一般的な動詞（例：「教えて」「知りたい」）は完全に無視します。
3.  固有名詞（例：「オーダーノート」）や、具体的な行動や方法を示す名詞（例：「書き方」「作り方」「手順」）を最優先します。
4.  抽象的な概念（例：「幸せ」「豊かさ」「周波数」）は、それが質問の明確な主題でない限り、含めないでください。
5.  抽出したキーワードを、重要度が高い順に半角スペースで区切って出力します。他の余計なテキストは一切含めないでください。

**これまでの会話履歴:**
{history_text}

**ユーザーの最新プロンプト:**
{prompt}

**生成された検索クエリ:**
"""
    
    response = model.generate_content(prompt_template)
    return response.text.strip()


@st.cache_resource(show_spinner=False)
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
        
        with st.spinner("宇宙と接続中だよ！ちょっとまってね..."):
            search_query = generate_search_query(prompt, st.session_state.messages)
            # 検索するドキュメント数を減らし、より関連性の高いものに絞る
            docs_with_scores = db.similarity_search_with_score(search_query, k=3)
            
            context_docs = []
            if docs_with_scores:
                # 関連性のスコア基準を厳しくする (より小さい値がより関連性が高い)
                context_docs = [doc for doc, score in docs_with_scores if score < 0.7]

            # 関連ドキュメントをセッション状態に保存
            if context_docs:
                sources = [
                    {
                        "file_path": doc.metadata.get("source", "不明"),
                        "score": score,
                    }
                    for doc, score in context_docs
                ]
            
            system_prompt = f"""
---
## AIの応答に関する指示 (着眼点シフトモード)
- **君の役割:**
  - 君の役割は、ユーザーの悩みを直接的に解決することではありません。
  - その悩みが、「オーダーノート」の哲学全体から見ると、どのような**「素晴らしい機会」**や**「成長のサイン」**に見えるか、その**新しい「着眼点」**を提示し、ユーザーの視点を180度転換させることが、君の唯一の役割です。

- **思考プロセス:**
  1.  ユーザーの悩み（例：お金がない、人間関係が悪い）の表面的な事象を受け取ります。
  2.  次に、その事象の裏にある**本質的なテーマ**（例：価値の受け取り方、自己肯定感、理想の世界観）を、君が持つナレッジベース全体から見抜きます。
  3.  そして、そのテーマに基づいて、ユーザーに**本質的な問い**を投げかけます。

- **具体的な会話開始の例:**
  - **ユーザーの悩み:** 「今、お金がピンチなんです！」
  - **君の応答（悪い例）:** 「大変ですね。節約する方法や、収入を増やす方法を考えてみましょう。」
  - **君の応答（良い例）:** 「そっか、今、お金という形で、君にパワフルなメッセージが届いているんだね。そのピンチは、君が『自分には価値がない』って無意識に握りしめている古い思い込みを、手放すための最高のチャンスかもしれないよ。もし、そのピンチが『君の本当の価値に気づけ！』っていう宇宙からのサインだとしたら、何から始めてみたい？」
---
関連情報:
{context_docs if context_docs else "関連情報なし"}
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