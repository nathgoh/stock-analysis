import os
from datetime import datetime

from alpaca.data.historical import NewsClient
from alpaca.data.live import NewsDataStream
from alpaca.data.models import NewsSet
from alpaca.data.requests import NewsRequest
from dotenv import load_dotenv
from kafka import KafkaProducer

from text_utils import clean_text

load_dotenv()

ALPACA_API_KEY: str | None = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY: str | None = os.getenv("ALPACA_SECRET_KEY")

rest_client = NewsClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
request_params = NewsRequest(
    symbols="AAPL",
    start=datetime.strptime("2026-07-01", "%Y-%m-%d"),
    end=datetime.strptime("2026-07-26", "%Y-%m-%d"),
    include_content=True,
    exclude_contentless=False,
    sort="asc",
    limit=1000,
)

news: NewsSet = rest_client.get_news(request_params)


news_producer = KafkaProducer(
    bootstrap_servers="localhost:19092",
    # key_serializer=str.encode,
    # value_serializer=lambda v: v.encode('utf-8')
)

topic = "stock-news"


def on_success(metadata) -> None:
    print(f"Message produced to topic '{metadata.topic}' at offset {metadata.offset}")


def on_error(e) -> None:
    print(f"Error sending message: {e}")


# async def on_news(news_article) -> None:
#   print(news_article)


def news_stream(symbols: list[str]) -> None:
    stream = NewsDataStream(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
    )

    stream.subscribe_news(on_news, *symbols)
    stream.run()


def on_news(news_article) -> None:
    headline = news_article.headline
    summary = news_article.summary
    content = clean_text(news_article.content)
    date = news_article.created_at

    print(content)


on_news(news.data["news"][2])
