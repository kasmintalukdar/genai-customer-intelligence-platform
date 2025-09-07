import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="GenAI Customer Intelligence",
    page_icon="🤖",
    layout="wide"
)

# --- MODEL LOADING ---
# Caching the models is crucial for performance.
# Streamlit's cache decorators ensure that the models are loaded only once.

@st.cache_resource
def load_sentiment_pipeline():
    """Loads the sentiment analysis pipeline."""
    print("Loading Sentiment Analysis model...")
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_summarizer_pipeline():
    """Loads the summarization pipeline."""
    print("Loading Summarization model...")
    return pipeline("summarization", model="facebook/bart-large-cnn")

# Load the models using the cached functions
sentiment_pipeline = load_sentiment_pipeline()
summarizer_pipeline = load_summarizer_pipeline()


# --- DATA LOADING ---
@st.cache_data
def load_data(filepath):
    """Loads the review data from a CSV file."""
    print("Loading data...")
    df = pd.read_csv(filepath)
    # Basic cleaning: drop rows with missing reviews or product IDs
    df.dropna(subset=['Text', 'ProductId'], inplace=True)
    return df

df = load_data('data/reviews.csv')


# --- UI LAYOUT ---
st.title("🤖 GenAI Customer Intelligence Platform")
st.markdown("An interactive dashboard to analyze customer reviews using sentiment analysis and AI-powered summarization.")

# --- SIDEBAR FILTERS ---
st.sidebar.header("Product Selection")
# Use the 'ProductId' column for product selection
product_list = sorted(df['ProductId'].unique())
selected_product = st.sidebar.selectbox("Select a Product ID to Analyze", product_list)

# --- MAIN PANEL ---
st.header(f"Analysis for Product ID: {selected_product}")

# Filter the dataframe based on the selected product
product_df = df[df['ProductId'] == selected_product].copy()

if product_df.empty:
    st.warning("No reviews found for this product.")
else:
    # --- METRICS & SENTIMENT ANALYSIS ---
    total_reviews = len(product_df)
    
    # Apply sentiment analysis to each review (this can take a moment for many reviews)
    # Using a progress bar for better user experience
    st.write("Performing sentiment analysis...")
    progress_bar = st.progress(0)
    sentiments = []
    # Use the correct 'Text' column for the review content
    for i, review in enumerate(product_df['Text']):
        # Ensure review is a string before processing
        if isinstance(review, str):
            # Truncate review to avoid model errors with very long texts
            result = sentiment_pipeline(review[:512])[0] 
            sentiments.append(result['label'])
        else:
            # Append a neutral or placeholder label if the review is not text
            sentiments.append('NEUTRAL') # Or handle as you see fit
        progress_bar.progress((i + 1) / total_reviews)
    
    product_df['sentiment'] = sentiments

    # Calculate sentiment distribution
    sentiment_counts = product_df['sentiment'].value_counts()
    positive_reviews = sentiment_counts.get("POSITIVE", 0)
    negative_reviews = sentiment_counts.get("NEGATIVE", 0)
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", f"{total_reviews}")
    col2.metric("Positive Reviews", f"{positive_reviews}", f"{positive_reviews/total_reviews:.2%}" if total_reviews > 0 else "0%")
    col3.metric("Negative Reviews", f"{negative_reviews}", f"-{negative_reviews/total_reviews:.2%}" if total_reviews > 0 else "0%")
    
    # --- VISUALIZATION ---
    st.subheader("Sentiment Distribution")
    if not sentiment_counts.empty:
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90,
                              colors=['#4CAF50', '#F44336', '#FFC107']) # Added yellow for neutral
        ax.set_ylabel('') # Hide the 'sentiment' label on the y-axis
        st.pyplot(fig)
    else:
        st.write("No sentiment data to display.")
    
    # --- AI-POWERED SUMMARY ---
    st.subheader("AI-Powered Summary of Negative Reviews")
    # Use the correct 'Text' column
    negative_review_texts = product_df[product_df['sentiment'] == 'NEGATIVE']['Text'].tolist()

    if not negative_review_texts:
        st.info("No negative reviews to summarize for this product.")
    else:
        with st.spinner("Generating summary... This may take a minute."):
            # Ensure all items in the list are strings
            negative_review_texts = [str(r) for r in negative_review_texts]
            full_negative_text = " ".join(negative_review_texts)
            # Truncate the combined text to fit within the model's limit
            summary = summarizer_pipeline(full_negative_text[:1024], max_length=150, min_length=40, do_sample=False)[0]['summary_text']
            st.success("Summary Generated!")
            st.write(summary)

    # --- RAW DATA ---
    with st.expander("View Raw Reviews"):
        # Use the correct column names 'Text' and 'Score'
        st.dataframe(product_df[['Text', 'sentiment', 'Score']].rename(
            columns={'Text': 'Review', 'sentiment': 'AI Sentiment', 'Score': 'Star Rating'}
        ))


