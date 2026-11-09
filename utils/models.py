from datetime import datetime

from pydantic import BaseModel


class NewsMessage(BaseModel):
    """
    One Alpaca news article. Sent once per article (not per symbol), keyed by `id`,
    so consumers can dedupe redelivered or backfilled articles.
    """

    id: int
    headline: str
    summary: str
    content: str
    source: str
    url: str | None
    author: str
    # Symbols the article is about, as tagged by Alpaca (may be empty).
    symbols: list[str]
    created_at: datetime
    updated_at: datetime

    def to_kafka_value(self) -> str:
        return self.model_dump_json()
