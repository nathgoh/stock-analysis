from alpaca.data.live import NewsDataStream
from alpaca.data.models import News
from kafka import KafkaProducer

from .messages import alpaca_credentials, send_article


def news_stream(redpanda_client: KafkaProducer, topic: str, symbols: list[str]) -> None:
    """
    Stream live news for `symbols` ("*" for all news), one message per article.
    """

    stream = NewsDataStream(*alpaca_credentials())

    async def on_news(news_article: News | dict) -> None:
        # Articles only arrive as dicts with `raw_data=True`.
        assert isinstance(news_article, News)
        send_article(redpanda_client, topic, news_article)
        print(
            f"Streamed article {news_article.id} {news_article.symbols}: "
            f"{news_article.headline}",
            flush=True,
        )

    stream.subscribe_news(on_news, *symbols)
    stream.run()
