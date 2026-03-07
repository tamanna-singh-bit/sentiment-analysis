import streamlit as st
from senti import predict  # your existing sentiment function

def main():
    st.set_page_config(page_title="Product Review Sentiment Analysis", layout="centered")
    st.title("📊 Product Review Sentiment Analysis")
    st.write("Type a product review below and click **Analyze** to see its sentiment.")

    review = st.text_area("Your Review:", height=150, placeholder="Enter review text here")

    if st.button("Analyze"):
        if not review.strip():
            st.warning("Please enter some text to analyze.")
        else:
            sentiment, confidence = predict(review)
            if sentiment == "positive":
                st.success(f"✅ Sentiment: {sentiment.upper()} — Confidence: {confidence}%")
            else:
                st.error(f"❌ Sentiment: {sentiment.upper()} — Confidence: {confidence}%")

if __name__ == "__main__":
    main()
