from dotenv import load_dotenv
load_dotenv()

import os

import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI


# =========================
# LLM呼び出し用の関数
# =========================
def ask_expert(user_text: str, expert_type: str) -> str:
    """
    入力テキストと専門家タイプを元に LLM へ問い合わせ、
    回答テキストを返す関数。

    Parameters
    ----------
    user_text : str
        ユーザーが入力フォームから送信したテキスト
    expert_type : str
        ラジオボタンで選択された専門家の種類

    Returns
    -------
    str
        LLM からの回答テキスト
    """

    # 専門家タイプごとにシステムメッセージを切り替え
    if expert_type == "キャリアコーチ（仕事・転職の相談）":
        system_prompt = (
            "あなたはプロのキャリアコーチです。"
            "相談者の経歴や価値観を丁寧に聞き出しながら、"
            "日本のビジネス文化を踏まえた現実的なアドバイスを日本語で行ってください。"
            "専門用語を使いすぎず、具体的な次の一歩が分かる提案をしてください。"
        )
    elif expert_type == "ライフプランナー（お金・将来設計の相談）":
        system_prompt = (
            "あなたはファイナンシャルプランナーの専門家です。"
            "相談者のライフイベント（結婚、出産、教育、老後など）を意識しながら、"
            "貯蓄や投資、保険などについて中立的な立場で日本語で説明してください。"
            "具体例と注意点を分かりやすく示し、行動に移しやすい形で助言してください。"
        )
    else:
        # 念のためデフォルト
        system_prompt = "あなたは丁寧で親切な日本語のアシスタントです。"

    # OpenAI APIキーの取得（st.secrets → 環境変数 の順で参照）
    api_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI APIキーが設定されていません。\n"
            "Streamlit Community Cloud の『Secrets』もしくは"
            "環境変数 OPENAI_API_KEY にキーを設定してください。"
        )

    # LangChain の LLM インスタンス生成
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # お好みで変更可
        api_key=api_key,
        temperature=0.7,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ]

    # LLM 呼び出し
    response = llm.invoke(messages)
    return response.content


# =========================
# Streamlit アプリ本体
# =========================
def main():
    st.set_page_config(
        page_title="専門家チャットデモ",
        page_icon="💬",
    )

    st.title("💬 専門家チャットWebアプリ")

    # アプリの概要・操作方法
    st.markdown(
        """
### アプリの概要

このアプリは、**LangChain** と **OpenAI の LLM** を利用したシンプルなチャットデモです。

- 画面上の **入力フォーム** から質問や相談内容を入力すると、
- 選択した **専門家の視点** で LLM が回答を生成し、
- 結果がこのページ上に表示されます。

### 使い方

1. 下のラジオボタンから、LLMに振る舞ってほしい **専門家の種類** を選びます。  
2. テキストエリアに、相談したい内容や聞きたいことを自由に入力します。  
3. 「送信」ボタンを押すと、LLM が選択した専門家として回答を返します。  
"""
    )

    # 専門家の種類（ラジオボタン）
    expert_type = st.radio(
        "LLM にどの専門家として振る舞ってほしいですか？",
        options=[
            "キャリアコーチ（仕事・転職の相談）",
            "ライフプランナー（お金・将来設計の相談）",
        ],
        horizontal=False,
    )

    # 入力フォーム
    user_text = st.text_area(
        "相談内容・質問を入力してください：",
        height=150,
        placeholder="例）IT業界から別の業界へ転職したいのですが、何から始めると良いでしょうか？",
    )

    # 送信ボタン
    if st.button("送信"):
        if not user_text.strip():
            st.warning("相談内容が空です。テキストを入力してください。")
        else:
            try:
                with st.spinner("専門家が考えています..."):
                    answer = ask_expert(user_text, expert_type)

                st.markdown("### 🔍 専門家からの回答")
                st.write(answer)

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
