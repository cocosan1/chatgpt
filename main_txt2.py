import streamlit as st
from llama_index import (
    GPTVectorStoreIndex,
    ServiceContext,
    StorageContext,
    SimpleDirectoryReader,
    Document,
    GPTListIndex,
    load_index_from_storage
)
from llama_index.prompts.prompts import RefinePrompt, QuestionAnswerPrompt
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore

import os
import glob
import sys

from google.oauth2 import service_account
from googleapiclient.discovery import build

#環境変数にキーをセット
os.environ['OPENAI_API_KEY'] == st.secrets['OPENAI_API_KEY']

st.markdown('### chatgpt Q&A')

def make_index():
##########################google driveからテキストファイルの取得
    #current working dir
    cwd = os.path.dirname(__file__)

    #**********************gdriveからテキストファイルのダウンロード・df化

    # Google Drive APIを使用するための認証情報を取得する
    creds_dict = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(creds_dict)

    # Drive APIのクライアントを作成する
    #API名（ここでは"drive"）、APIのバージョン（ここでは"v3"）、および認証情報を指定
    service = build("drive", "v3", credentials=creds)

    #フォルダtexts内にあるファイルリストを作成
    folder_id = st.secrets["ID"]  # 取得したいフォルダのIDを指定してください

    # フォルダ内のファイルを取得するためのクエリ
    query = f"'{folder_id}' in parents"

    # フォルダ内のファイル名を取得するリクエスト
    response = service.files().list(q=query, fields="files(name)").execute()

    # レスポンスからファイル名を取得する
    files = response.get("files", [])


    for fname in files:
        # 指定したファイル名を持つファイルのIDを取得する
        #Google Drive上のファイルを検索するためのクエリを指定して、ファイルの検索を実行します。
        # この場合、ファイル名とMIMEタイプを指定しています。

        file_name = fname["name"]
        query = f"name='{file_name}' and mimeType='text/plain'"


        #指定されたファイルのメディアを取得
        results = service.files().list(q=query).execute()
        items = results.get("files", [])

        if not items:
            st.caption(f"No files found with name: {file_name}")
        else:
            # ファイルをダウンロードする
            file_id = items[0]["id"]
            file = service.files().get(fileId=file_id).execute()
            file_content = service.files().get_media(fileId=file_id).execute()

            # ファイルを保存する
            file_path = os.path.join(cwd, 'texts', file_name)
            with open(file_path, "wb") as f:
                f.write(file_content)
        
    ###########################テキストデータの統合
    #検索するディレクトリを指定する
    directory = os.getcwd() #このPythonファイルが保存されているフォルダを指定

    #txt_files というリストに拡張子が.txtのファイルパスを入れる
    txt_files = glob.glob(directory + "/texts/*.txt")

    #リストが空の場合は終了する
    if not txt_files:
        input('拡張子が.txtのファイルが見つかりませんでした Enterで終了します')
        sys.exit() #プログラムを終了させる

    #結合後のファイル名を指定する
    merged_file_name = "./main/main.txt"

    # ファイルを開いて結合する------------------------------------------
    #結合後のファイルを開く(W:ファイルを書き込み用に開き中身を削除する。ファイルが存在しなければ作成)
    with open(merged_file_name, "w",encoding="utf-8") as merged_file:
        #リスト内の項目を1つずつ処理する
        for file_name in txt_files:

            #結合後のファイルは対象外にする
            if merged_file_name in file_name:
                continue

            #リスト内のファイルを読み取り専用で開く
            with open(file_name, "r",encoding="utf-8") as file:
                #read_textという変数にファイル内のすべてのデータを代入
                read_text = file.read()
                #もし最後が改行で終わっていない場合は改行を追加する
                if not read_text == '' and not read_text.endswith('\n'):
                        read_text = read_text + '\n'
                #結合後のファイルに書き込む
                merged_file.write(read_text)

    ###########################テキストデータの読み込み・index化
    documents = SimpleDirectoryReader(
        input_dir="./main").load_data()

    # GPTVectorStoreIndexの作成
    vector_store_index = GPTListIndex.from_documents(documents)

    ###########################persistでstorage_contextを保存
    storage_context = vector_store_index.storage_context
    storage_context.persist(persist_dir="./storage_context")

    st.info('index化完了')

    # response変数を初期化する
    response = None

def q_and_a():

    #質問の入力
    question = st.text_input('質問を入力してください')

    if not question:
        st.info('質問を入力してください')
        st.stop()
    #storage_contextの読み込み
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./storage_context"),
        vector_store=SimpleVectorStore.from_persist_dir(persist_dir="./storage_context"),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir="./storage_context"),
    )

    #############################Q&A
    # インデックスの読み込み
    index = load_index_from_storage(storage_context)

    # query_engine = index.as_query_engine()
    # # 質問のテンプレートにコンテキストと質問を埋め込む
    # context_str = "\n".join(response.source_nodes)
    # query_str = question

    # QA_PROMPT_TMPL = (
    # "私たちは以下の情報をコンテキスト情報として与えます。 \n"
    # "---------------------\n"
    # "{context_str}"
    # "\n---------------------\n"
    # "あなたはAIとして、この情報をもとに質問を日本語で答えます。前回と同じ回答の場合は同じ回答を行います。: {query_str}\n"
    # )

    # prompt = QA_PROMPT_TMPL.format(context_str=context_str, query_str=query_str)

    # # 質問を行う
    # response = query_engine.query(prompt)


    #基本形
    query_engine = index.as_query_engine()
    response = query_engine.query(question)

    for i in response.response.split("。"):
        st.write(i + "。")


    # #ソースの表示
    # st.write(response.source_nodes)

def main():
    # アプリケーション名と対応する関数のマッピング
    apps = {
        'Q&A': q_and_a,
        'txtのindex化': make_index

    }
    selected_app_name = st.selectbox(label='項目の選択',
                                             options=list(apps.keys()))
    

    # 選択されたアプリケーションを処理する関数を呼び出す
    render_func = apps[selected_app_name]
    render_func()

if __name__ == '__main__':
    main()





