import json
import os
from datetime import UTC, datetime, timedelta

from alpaca.data.historical import NewsClient
from alpaca.data.live import NewsDataStream
from alpaca.data.models import News, NewsSet
from alpaca.data.requests import NewsRequest
from dotenv import load_dotenv
from kafka import KafkaProducer

from text_utils import clean_text

load_dotenv()

ALPACA_API_KEY: str | None = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY: str | None = os.getenv("ALPACA_SECRET_KEY")


def on_success(metadata) -> None:
    print(f"Message produced to topic '{metadata.topic}' at offset {metadata.offset}")


def on_error(e) -> None:
    print(f"Error sending message: {e}")


# def news_stream(symbols: list[str]) -> None:
#     stream = NewsDataStream(
#         api_key=ALPACA_API_KEY,
#         secret_key=ALPACA_SECRET_KEY,
#     )

#     stream.subscribe_news(on_news, *symbols)
#     stream.run()

# async def on_news(news_article) -> None:
#     headline: str = news_article.headline
#     summary: str = news_article.summary
#     content: str = clean_text(news_article.content)
#     date: datetime = news_article.created_at


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

        headline: str = news_article.headline
        summary: str = news_article.summary
        content: str = clean_text(news_article.content)
        date: datetime = news_article.created_at

        message = {
            "headline": headline,
            "summary": summary,
            "content": content,
            "date": date.isoformat(),
        }

        redpanda_client.send(
            topic,
            value=json.dumps(message),
            key=symbol,
            timestamp_ms=int(now.timestamp() * 1000),
        )
        redpanda_client.flush()


if __name__ == "__main__":
    topic = "market-news"
    news_producer = KafkaProducer(
        bootstrap_servers="localhost:19092",
        key_serializer=str.encode,
        value_serializer=lambda v: v.encode("utf-8"),
    )
    produce_historical_news(
        news_producer, topic, ["AAPL", "GOOG", "MSFT", "TSLA", "NVDA"]
    )

    news_producer.close()
