import os
from datetime import UTC, datetime, timedelta

from alpaca.data.historical import NewsClient
from alpaca.data.models import News, NewsSet
from alpaca.data.requests import NewsRequest
from dotenv import load_dotenv
from kafka import KafkaProducer

from utils.models import NewsMessage
from utils.text_utils import clean_text

load_dotenv()

ALPACA_API_KEY: str | None = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY: str | None = os.getenv("ALPACA_SECRET_KEY")


def on_success(metadata) -> None:
    print(f"Message produced to topic '{metadata.topic}' at offset {metadata.offset}")


def on_error(e) -> None:
    print(f"Error sending message: {e}")


def produce_historical_news(
    redpanda_client: KafkaProducer, topic: str, symbols: list[str]
) -> None:
    rest_client = NewsClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    for symbol in symbols:
        now = datetime.now(tz=UTC)
        request_params = NewsRequest(
            symbols=symbol,
            start=now - timedelta(days=365),
            end=now,
            include_content=True,
            exclude_contentless=True,
            sort="asc",
            limit=6000,
        )
        news: NewsSet = rest_client.get_news(request_params)
        news_article: News = news.data["news"][2]

        message = NewsMessage(
            headline=news_article.headline,
            summary=news_article.summary,
            content=clean_text(news_article.content),
            date=news_article.created_at,
        )

        future = redpanda_client.send(
            topic,
            value=message.to_kafka_value(),
            key=symbol,
            timestamp_ms=int(now.timestamp() * 1000),
        )
        future.add_callback(on_success)
        future.add_errback(on_error)
