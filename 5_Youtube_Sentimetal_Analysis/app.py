import gradio as gr
import requests
from googleapiclient.discovery import build
from transformers import pipeline
import re

# Setup Sentiment Analysis
sentiment_analyzer = pipeline("sentiment-analysis")

# YouTube API Key
YOUTUBE_API_KEY = "AIzaSyBuNxsm0LnHF0OkbYgMSNHnwu8iVUVi5gc"

def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def get_youtube_comments(video_url, max_results=20):
    video_id = extract_video_id(video_url)
    if not video_id:
        return None, "Invalid YouTube URL."

    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        order="relevance",
        textFormat="plainText"
    )
    response = request.execute()

    comments_data = []
    for item in response["items"]:
        comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        like_count = item["snippet"]["topLevelComment"]["snippet"].get("likeCount", 0)
        comments_data.append((comment_text, like_count))

    return comments_data, None

def analyze_comments(video_url):
    comments_data, error = get_youtube_comments(video_url)
    if error:
        return error

    table_md = "| Comment | Sentiment | Likes |\n|---|---|---|\n"
    summary_data = {
        "positive": 0,
        "neutral": 0,
        "negative": 0,
        "max_likes": -1,
        "top_comment": "",
        "top_sentiment": "",
        "most_positive": "",
        "most_negative": ""
    }

    for comment, likes in comments_data:
        result = sentiment_analyzer(comment)[0]
        sentiment = result["label"]

        if sentiment == "POSITIVE":
            summary_data["positive"] += 1
        elif sentiment == "NEGATIVE":
            summary_data["negative"] += 1
        else:
            summary_data["neutral"] += 1

        table_md += f"| {comment} | {sentiment} | {likes} |\n"

        if likes > summary_data["max_likes"]:
            summary_data["max_likes"] = likes
            summary_data["top_comment"] = comment
            summary_data["top_sentiment"] = sentiment

        if sentiment == "POSITIVE":
            summary_data["most_positive"] = comment
        if sentiment == "NEGATIVE":
            summary_data["most_negative"] = comment

    summary = (
        f"\n\n### Summary:\n"
        f"- Most liked comment: \"{summary_data['top_comment']}\" ({summary_data['max_likes']} likes, {summary_data['top_sentiment']})\n"
        f"- Most positive comment: \"{summary_data['most_positive']}\"\n"
        f"- Most negative comment: \"{summary_data['most_negative']}\"\n"
        f"- Sentiment Count: {summary_data['positive']} Positive, {summary_data['neutral']} Neutral, {summary_data['negative']} Negative\n"
    )

    return table_md + summary

interface = gr.Interface(
    fn=analyze_comments,
    inputs=gr.Textbox(label="Enter the Youtube Link:"),
    outputs=gr.Markdown(),
    title="YouTube Comment Sentiment Analyzer",
    description="Paste a YouTube video link to analyze top comments for sentiment (Positive, Negative, Neutral)."
)

interface.launch()
