import json
import os
from datetime import datetime

from alpaca.data.live import NewsDataStream
from dotenv import load_dotenv
from kafka import KafkaProducer

from utils.text_utils import clean_text

load_dotenv()

ALPACA_API_KEY: str | None = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY: str | None = os.getenv("ALPACA_SECRET_KEY")

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
        content: str = clean_text(news_article.content)
        news_date: datetime = news_article.created_at
        message = {
            "headline": news_article.headline,
            "summary": news_article.summary,
            "content": content,
            "date": news_date.isoformat(),
        }

        for symbol in news_article.symbols:
            redpanda_client.send(
                topic,
                key=symbol,
                value=json.dumps(message)
            )

    stream.subscribe_news(on_news, *symbols)
    stream.run()