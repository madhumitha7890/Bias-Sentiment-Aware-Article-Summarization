import os
import streamlit as st
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import biased_model
load_dotenv()
# Read token from environment variable
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    st.error("GITHUB_TOKEN not found. Please set it as an environment variable.")
    st.stop()

# Initialize GPT-4.1 client
endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4.1"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(GITHUB_TOKEN),
)

# Streamlit UI
st.set_page_config(page_title="News Article Analyzer", layout="wide")
st.title("📰 News Article Summarizer, Sentiment & Bias Detector")

article_text = st.text_area("Paste your news article here:", height=300)

if st.button("Analyze"):
    if not article_text.strip():
        st.warning("Please enter some article text.")
    else:
        with st.spinner("Generating summary with GPT-4.1..."):
            try:
                user_prompt = (
                    "Please summarize the following news article in a concise and clear manner:\n\n"
                    f"{article_text}"
                )

                response = client.complete(
                    messages=[
                        SystemMessage("You are a helpful assistant that summarizes news articles."),
                        UserMessage(user_prompt)
                    ],
                    temperature=0.7,
                    top_p=1,
                    model=model_name
                )

                summary = response.choices[0].message.content.strip()
                st.subheader("📝 Summary (GPT-4.1)")
                st.success(summary)

            except Exception as e:
                st.error(f"Failed to generate summary: {e}")
                st.stop()

        with st.spinner("Analyzing sentiment..."):
            sentiment = biased_model.get_final_sentiment(article_text)
            st.subheader("❤️ Sentiment")
            st.info(f"Detected sentiment: **{sentiment}**")

        with st.spinner("Detecting bias..."):
            bias_scores = biased_model.detect_bias(article_text)
            st.subheader("⚖️ Bias Detection")
            st.json(bias_scores)
