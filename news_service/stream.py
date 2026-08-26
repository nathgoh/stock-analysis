import os

from alpaca.data.live import NewsDataStream
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


def news_stream(redpanda_client: KafkaProducer, topic: str, symbols: list[str]) -> None:
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise ValueError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables must be set."
        )

    stream = NewsDataStream(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
    )

    async def on_news(news_article) -> None:
        message = NewsMessage(
            headline=news_article.headline,
            summary=news_article.summary,
            content=clean_text(news_article.content),
            date=news_article.created_at,
        )

        for symbol in news_article.symbols:
            future = redpanda_client.send(
                topic,
                key=symbol,
                value=message.to_kafka_value(),
            )
            future.add_callback(on_success)
            future.add_errback(on_error)

    stream.subscribe_news(on_news, *symbols)
    stream.run()
