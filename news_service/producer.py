from datetime import UTC, datetime, timedelta

from alpaca.data.historical import NewsClient
from alpaca.data.models import NewsSet
from alpaca.data.requests import NewsRequest
from kafka import KafkaProducer

from .messages import alpaca_credentials, send_article

# Days of news to backfill when no `backfill_days` is configured.
DEFAULT_BACKFILL_DAYS = 7


def produce_historical_news(
    redpanda_client: KafkaProducer,
    topic: str,
    symbols: list[str],
    backfill_days: int = DEFAULT_BACKFILL_DAYS,
) -> None:
    """
    Send every article of the last `backfill_days` days for `symbols` ("*" for all
    news), oldest first. Fetched one day at a time: the SDK loads all pages of a
    request (50 articles each) into memory before returning.
    """

    rest_client = NewsClient(*alpaca_credentials())
    # The REST API has no "*" wildcard (only the live stream does), leaving the symbols
    # out returns all news.
    symbol_filter = None if "*" in symbols else ",".join(symbols)

    end = datetime.now(tz=UTC)
    day_start = end - timedelta(days=backfill_days)
    total = 0
    while day_start < end:
        day_end = min(day_start + timedelta(days=1), end)
        request_params = NewsRequest(
            symbols=symbol_filter,
            start=day_start,
            end=day_end,
            include_content=True,
            # Many short news items (e.g. analyst actions) have no body, the headline is
            # what gets scored.
            exclude_contentless=False,
            sort="asc",
        )
        news = rest_client.get_news(request_params)
        assert isinstance(news, NewsSet)

        articles = news.data.get("news", [])
        for article in articles:
            send_article(redpanda_client, topic, article)
        # Flush per day, so an interrupted backfill has sent every finished day.
        redpanda_client.flush()

        total += len(articles)
        print(
            f"Backfilled {len(articles)} articles from {day_start:%Y-%m-%d %H:%M} UTC "
            f"({total} total)",
            flush=True,
        )
        day_start = day_end
