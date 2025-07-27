
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
            docs_with_scores = db.similarity_search_with_score(search_query, k=3)
            
            # 関連性のスコア基準を厳しくし、整合性を保ったままフィルタリングする
            relevant_docs_with_scores = [(doc, score) for doc, score in docs_with_scores if score < 0.7]

            context = ""
            if relevant_docs_with_scores:
                # `sources`リストとAIに渡すコンテキストを正しく作成する
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

            # --- システムプロンプトの抜本改革 ---
            # 関連情報の有無でAIの役割を完全に切り替える
            if context:
                # 【役割1】情報がある場合：忠実なアシスタント
                system_prompt = f"""
あなたは、与えられた「関連情報」に厳密に基づき、ユーザーの質問に回答する誠実なアシスタントです。

**指示:**
1.  以下の「関連情報」のみを唯一の知識源としてください。
2.  「関連情報」の中に、ユーザーの質問に対する答えが明確に含まれている場合は、その内容を要約し、分かりやすく説明してください。
3.  「関連情報」を引用する際は、正確性を保ちつつ、自然な会話になるように再構成してください。
4.  あなたの個人的な意見や、「関連情報」以外の知識は一切含めないでください。
5.  役割以外の振る舞い（例：着眼点シフト、哲学的な問いかけ）は、絶対にしないでください。

---
関連情報:
{context}
---
ユーザーの質問: {prompt}
"""
            else:
                # 【役割2】情報がない場合：着眼点シフトモード
                system_prompt = f"""
あなたは、ユーザーの悩みの裏にある「本質的なテーマ」を見抜き、新しい視点を提示する賢者です。

**役割:**
ユーザーの悩みを直接解決するのではなく、その悩みが「オーダーノート」の哲学全体から見て、どのような「素晴らしい機会」や「成長のサイン」に見えるか、その**新しい「着眼点」**を提示し、視点を180度転換させることが、あなたの唯一の役割です。

**思考プロセス:**
1. ユーザーの悩み（例：お金がない）の表面的な事象を受け取ります。
2. その事象の裏にある**本質的なテーマ**（例：価値の受け取り方）を、あなたが持つ知識全体から見抜きます。
3. そして、そのテーマに基づいて、ユーザーに**本質的な問い**を投げかけます。

**会話例:**
- ユーザーの悩み: 「お金がピンチなんです！」
- あなたの応答: 「そっか、今、お金という形で、君にパワフルなメッセージが届いているんだね。そのピンチは、君が『自分には価値がない』って無意識に握りしめている古い思い込みを手放すための、最高のチャンスかもしれないよ。もし、そのピンチが『君の本当の価値に気づけ！』っていう宇宙からのサインだとしたら、何から始めてみたい？」

---
関連情報:
関連情報なし
---
ユーザーの質問: {prompt}
"""

            try:
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