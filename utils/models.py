from datetime import datetime

from pydantic import BaseModel


class NewsMessage(BaseModel):
    headline: str
    summary: str
    content: str
    date: datetime

    def to_kafka_value(self) -> str:
        return self.model_dump_json()
