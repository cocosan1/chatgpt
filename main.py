import os
from pathlib import Path
from dotenv import load_dotenv #環境変数を読み込む
import streamlit as st
from streamlit_chat import message #チャット履歴の表示がリッチ

from llama_index import (
    download_loader,
    LLMPredictor, #LLMPredictorはテキスト応答（Completion）を得るための言語モデルの部分を担っています。
    GPTVectorStoreIndex, 
    #各Nodeに対応する埋め込みベクトルと共に順序付けせずに保持。
    # 埋め込みベクトルを使用してNodeを抽出し、それぞれの出力を合成
    ServiceContext,
    QuestionAnswerPrompt, #コンテキストに対して回答をもとめるようなプロンプト
    StorageContext,
    load_index_from_storage
)
from langchain import OpenAI

#環境変数を読み込む
load_dotenv()

#フォルダの設定と生成
PDF_DATA_DIR = "./pdf_data/"
STORAGE_DIR = "./storage/"
os.makedirs(PDF_DATA_DIR, exist_ok=True)

#PDF読み込み
class PDFReader:
    def __init__(self):
        self.pdf_reader = download_loader("PDFReader")()

    def load_data(self, file_name):
        #テキストファイルの情報を取得
        return self.pdf_reader.load_data(file=Path(PDF_DATA_DIR + file_name))

class QAResponseGenerator:
    def __init__(self, selected_model, pdf_reader):
        #モデル定義　LLMPredictorはテキスト応答（Completion）を得るための言語モデルの部分を担っている
        self.llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name=selected_model))
        self.pdf_reader = pdf_reader
        self.QA_PROMPT_TMPL = (
            "下記の情報が与えられています。 \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "この情報を参照して次の質問に答えてください: {query_str}\n"
        )
        #llama-indexがIndexを作ったりクエリを実行する際に必要になる部品をまとめる
        #gpt-3.5-turbo を指定（現状デフォルトは davinci ）
        self.service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor)

    def generate(self, question, file_name):
            documents = self.pdf_reader.load_data(file_name)
            try:
                ## Storage Contextの作成
                storage_context = StorageContext.from_defaults(persist_dir=f"{STORAGE_DIR}{file_name}")
                index = load_index_from_storage(storage_context)
                print("load existing file..")
            except:
                # documents をもとに Embbeddings API を通信してベクター取得し GPTSimpleVectorIndex を生成
                index = GPTVectorStoreIndex.from_documents(documents, service_context=self.service_context)
                index.storage_context.persist(persist_dir=f"{STORAGE_DIR}{file_name}")
            
            #Storage ContextやService Context以外の設定
            #QuestionAnswerPrompt コンテキストに対して回答をもとめるようなプロンプト
            engine = index.as_query_engine(text_qa_template=QuestionAnswerPrompt(self.QA_PROMPT_TMPL))
            # ベクター検索 + Chat Completion API 実行
            result = engine.query(question)

            #get_formatted_sources　gpt-3.5-turbo が回答を生成したときに一緒に送ったドキュメントを出力する機能
            return result.response.replace("\n", ""), result.get_formatted_sources(1000)

#PDFファイルを保存
def save_uploaded_file(uploaded_file, save_dir):
    try:
        with open(os.path.join(save_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getvalue())
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False

#ファイルアップロード
def upload_pdf_file():
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        #PDFファイルを保存
        if save_uploaded_file(uploaded_file, PDF_DATA_DIR):
            st.success(f"File {uploaded_file.name} saved at {PDF_DATA_DIR}")
        else:
            st.error("The file could not be saved.")

#チャットの履歴を表示
def display_chat(chat_history):
    for i, chat in enumerate(reversed(chat_history)):
        if "user" in chat:
            message(chat["user"], is_user=True, key=str(i)) #message チャット履歴の表示がリッチ
        else:
            message(chat["bot"], key="bot_"+str(i))

def main():
    st.title('PDF Q&A app')

    #ファイルアップロード
    upload_pdf_file()
    #ファイル選択
    file_name = st.sidebar.selectbox("Choose a file", os.listdir(PDF_DATA_DIR)) 
    #モデル選択
    selected_model = st.sidebar.selectbox("Choose a model", ["gpt-3.5-turbo", "gpt-4"])
    choice = st.radio("参照情報を表示:", ["表示する", "表示しない"])
    question = st.text_input("Your question")

    # メインの画面に質問送信ボタンを設定
    submit_question = st.button("質問")
    clear_chat = st.sidebar.button("履歴消去")

    # チャット履歴を保存
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if clear_chat:
        st.session_state["chat_history"] = []

    #PDF読み込み
    pdf_reader = PDFReader()
    response_generator = QAResponseGenerator(selected_model, pdf_reader)
    # ボタンがクリックされた場合の処理
    if submit_question:
        if question:  # 質問が入力されている場合
            response, source = response_generator.generate(question, file_name)
            if choice == "表示する":
                response += f"\n\n参照した情報は次の通りです:\n{source}"

            # 質問と応答をチャット履歴に追加
            st.session_state["chat_history"].append({"user": question})
            st.session_state["chat_history"].append({"bot": response})

    #チャットの履歴を表示
    display_chat(st.session_state["chat_history"])


if __name__ == "__main__":
    main()


