import os

from alpaca.data.models import News
from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.producer.future import FutureRecordMetadata

from utils.models import NewsMessage
from utils.text_utils import clean_text

load_dotenv()


def alpaca_credentials() -> tuple[str, str]:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables must be set."
        )

    return api_key, secret_key


def to_message(article: News) -> NewsMessage:
    return NewsMessage(
        id=article.id,
        headline=article.headline,
        summary=article.summary,
        content=clean_text(article.content),
        source=article.source,
        url=article.url,
        author=article.author,
        symbols=article.symbols,
        created_at=article.created_at,
        updated_at=article.updated_at,
    )


def on_error(e) -> None:
    print(f"Error sending message: {e}")


def send_article(
    redpanda_client: KafkaProducer, topic: str, article: News
) -> FutureRecordMetadata:
    """
    Send one article, keyed by its id and timestamped with its creation time.
    """

    message = to_message(article)
    future = redpanda_client.send(
        topic,
        key=str(message.id),
        value=message.to_kafka_value(),
        timestamp_ms=int(message.created_at.timestamp() * 1000),
    )
    future.add_errback(on_error)

    return future
