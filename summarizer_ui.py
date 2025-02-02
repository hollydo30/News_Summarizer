import requests
from bs4 import BeautifulSoup
from newspaper import Article
from transformers import pipeline
import gradio as gr

# Initialize the summarization pipeline using BART
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to scrape document from a URL
def get_document_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

# Preprocess the text to clean it
def preprocess_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ').strip()  # Remove newlines and extra spaces
    return text

# Function to summarize text
def summarize_text(text):
    max_input_length = 1024  # BART's max token limit
    text = text[:max_input_length]  # Truncate if necessary
    
    result = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return result[0]['summary_text']  

# Function to analyze document from a URL
def analyze_document(url):
    # Step 1: Scrape the document
    document_text = get_document_text(url)
    
    # Step 2: Preprocess the document text
    cleaned_text = preprocess_text(document_text)
    
    # Step 3: Summarize the document
    summary = summarize_text(cleaned_text)
    
    return summary

# Chatbot function
def chat_with_bot(url):
    summary = analyze_document(url)
    return f"Summary:\n{summary}"

# Gradio UI
demo = gr.Interface(
    fn=chat_with_bot,
    inputs=gr.Textbox(label="Enter URL"),
    outputs="text",
    title="News Summarizer"
)

# Launch local server
demo.launch()
