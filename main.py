import os
import streamlit as st
import uuid
from datetime import datetime
import json
from pathlib import Path

from langchain.document_loaders import TextLoader, CSVLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# from dotenv import load_dotenv
# load_dotenv()

# # 環境変数からAPIキーを読み込み
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# correct_username = os.getenv("LOGIN_USERNAME")
# correct_password = os.getenv("LOGIN_PASSWORD")
OPENAI_API_KEY = st.secrets("OPENAI_API_KEY")
GOOGLE_API_KEY = st.secrets("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = st.secrets("ANTHROPIC_API_KEY")
correct_username = st.secrets("LOGIN_USERNAME")
correct_password = st.secrets("LOGIN_PASSWORD")


# 定数
CHAT_HISTORY_FILE = "chat_history.json"
TEMP_DIR = "temp_uploads"
VECTOR_STORE_DIR = "vector_store"

# APIキーのチェック
def check_api_keys():
    missing_keys = []
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    if not GOOGLE_API_KEY:
        missing_keys.append("GOOGLE_API_KEY")
    if not ANTHROPIC_API_KEY:
        missing_keys.append("ANTHROPIC_API_KEY")
    
    if missing_keys:
        st.error(f"環境変数に次のAPIキーが設定されていません: {', '.join(missing_keys)}")
        return False
    return True

# ディレクトリが存在しない場合は作成
for directory in [TEMP_DIR, VECTOR_STORE_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# セッション状態の初期化
def init_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'chat_id' not in st.session_state:
        st.session_state.chat_id = str(uuid.uuid4())
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'current_title' not in st.session_state:
        st.session_state.current_title = "新しいチャット"
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# チャット履歴の読み込み
def load_chat_history():
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"チャット履歴の読み込みエラー: {e}")
        return {}

# チャット履歴の保存
def save_chat_history():
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"チャット履歴の保存エラー: {e}")

# 新しいチャットの作成
def create_new_chat():
    st.session_state.chat_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.uploaded_files = []
    st.session_state.current_title = "新しいチャット"
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 過去のチャットを読み込む
def load_chat(chat_id):
    if chat_id in st.session_state.chat_history:
        st.session_state.chat_id = chat_id
        st.session_state.messages = st.session_state.chat_history[chat_id]["messages"]
        st.session_state.current_title = st.session_state.chat_history[chat_id]["title"]
        st.session_state.uploaded_files = st.session_state.chat_history[chat_id].get("uploaded_files", [])
        
        # メモリの再構築
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                memory.chat_memory.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                memory.chat_memory.add_ai_message(msg["content"])
        st.session_state.memory = memory

# ファイルのアップロード処理
def process_uploaded_file(uploaded_file):
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    
    # ファイルを保存
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # ファイル拡張子に基づいてローダーを選択
    file_extension = Path(uploaded_file.name).suffix.lower()
    documents = []
    
    try:
        if file_extension == ".txt":
            loader = TextLoader(file_path)
            documents = loader.load()
        elif file_extension == ".xlsx":
            loader = UnstructuredExcelLoader(file_path)
            documents = loader.load()
        else:
            st.error(f"サポートされていないファイル形式です: {file_extension}")
            return None
        
        # テキスト分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_documents = text_splitter.split_documents(documents)
        
        # ベクトルストアの作成
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        vector_store = Chroma.from_documents(
            documents=split_documents,
            embedding=embeddings,
            persist_directory=os.path.join(VECTOR_STORE_DIR, st.session_state.chat_id)
        )
        
        # アップロードされたファイルをセッションに追加
        file_info = {
            "name": uploaded_file.name,
            "path": file_path,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.uploaded_files.append(file_info)
        
        return vector_store
    
    except Exception as e:
        st.error(f"ファイル処理エラー: {e}")
        return None

# LLMモデルの取得
def get_llm(model_name):
    if model_name == "gemini-2.0-flash-lite":
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.7, google_api_key=GOOGLE_API_KEY)
    elif model_name == "gpt-4o-mini":
        return ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=OPENAI_API_KEY)
    elif model_name == "claude-3-7-sonnet-latest":
        return ChatAnthropic(model_name="claude-3-7-sonnet-20240229", temperature=0.7, anthropic_api_key=ANTHROPIC_API_KEY)
    else:
        # デフォルトモデル
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.7, google_api_key=GOOGLE_API_KEY)

# チャットの応答を生成
def generate_response(user_input, model_name, uploaded_files):
    try:
        llm = get_llm(model_name)
        
        if uploaded_files:
            # ドキュメントがある場合、RAGを使用
            vector_store = None
            for file_info in uploaded_files:
                file_path = file_info["path"]
                if os.path.exists(file_path):
                    file_extension = Path(file_path).suffix.lower()
                    documents = []
                    
                    if file_extension == ".txt":
                        loader = TextLoader(file_path)
                        documents = loader.load()
                    elif file_extension == ".xlsx":
                        loader = UnstructuredExcelLoader(file_path)
                        documents = loader.load()
                    
                    if documents:
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        split_documents = text_splitter.split_documents(documents)
                        
                        embeddings = GoogleGenerativeAIEmbeddings(
                            model="models/embedding-001",
                            google_api_key=GOOGLE_API_KEY
                        )
                        if vector_store is None:
                            vector_store = Chroma.from_documents(
                                documents=split_documents,
                                embedding=embeddings,
                                persist_directory=os.path.join(VECTOR_STORE_DIR, st.session_state.chat_id)
                            )
                        else:
                            vector_store.add_documents(split_documents)
            
            if vector_store:
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vector_store.as_retriever(),
                    memory=st.session_state.memory
                )
                response = qa_chain.invoke({"question": user_input})
                return response["answer"]
        
        # 通常のチャット応答（ファイルがない場合）
        prompt = f"すべての回答を日本語で返してください。質問: {user_input}"
        response = llm.invoke(prompt)
        return response.content
    
    except Exception as e:
        return f"エラーが発生しました: {e}"

# ログイン画面
def login_page():
    st.title("チャットシステムログイン")
    
    with st.form("login_form"):
        username = st.text_input("ユーザーID")
        password = st.text_input("パスワード", type="password")
        submit = st.form_submit_button("ログイン")
        
        if submit:
            # 環境変数からログイン情報を取得
            correct_username = os.environ.get("LOGIN_USERNAME")
            correct_password = os.environ.get("LOGIN_PASSWORD")
            
            if username == correct_username and password == correct_password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("ログインに失敗しました")

# チャット画面
def chat_page():
    st.title("LLMチャットシステム")
    
    # サイドメニュー
    with st.sidebar:
        st.button("新しいチャット", on_click=create_new_chat)
        
        st.subheader("過去のチャット履歴")
        for chat_id, chat_data in st.session_state.chat_history.items():
            if st.button(chat_data["title"], key=f"chat_{chat_id}"):
                load_chat(chat_id)
                st.rerun()

        st.write("---")
        if st.button("ログアウト"):
            st.session_state.authenticated = False
            st.session_state.page = 1
            st.session_state.user_input = ""
            st.rerun()        
    
    # チャット表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # チャット入力領域
    col1, col2, col3 = st.columns([0.15, 0.15, 0.7])
    
    with col1:
        uploaded_file = st.file_uploader("📎", type=["txt", "xlsx"], label_visibility="collapsed")
        if uploaded_file:
            vector_store = process_uploaded_file(uploaded_file)
            if vector_store:
                st.success(f"{uploaded_file.name}がアップロードされました")
    
    with col2:
        model_options = ["gemini-2.0-flash-lite", "gpt-4o-mini", "claude-3-7-sonnet-latest"]
        selected_model = st.selectbox("🤖", model_options, label_visibility="collapsed")
    
    # チャット入力
    if prompt := st.chat_input("メッセージを入力してください..."):
        # ユーザーメッセージの追加
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 応答の生成
        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
                response = generate_response(prompt, selected_model, st.session_state.uploaded_files)
                st.markdown(response)
        
        # アシスタントメッセージの追加
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # チャット履歴の更新
        if st.session_state.chat_id not in st.session_state.chat_history:
            # 最初のメッセージをタイトルとして使用
            title = prompt[:30] + "..." if len(prompt) > 30 else prompt
            st.session_state.current_title = title
            st.session_state.chat_history[st.session_state.chat_id] = {
                "title": title,
                "created_at": datetime.now().isoformat(),
                "messages": st.session_state.messages,
                "uploaded_files": st.session_state.uploaded_files
            }
        else:
            st.session_state.chat_history[st.session_state.chat_id]["messages"] = st.session_state.messages
            st.session_state.chat_history[st.session_state.chat_id]["uploaded_files"] = st.session_state.uploaded_files
        
        # チャット履歴を保存
        save_chat_history()

# メイン関数
def main():
    st.set_page_config(page_title="チャットシステム", layout="wide")
    init_session_state()
    
    # APIキーのチェック
    if not check_api_keys() and st.session_state.authenticated:
        st.warning("一部のAPIキーが設定されていないため、一部の機能が動作しない可能性があります")
    
    if not st.session_state.authenticated:
        login_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()