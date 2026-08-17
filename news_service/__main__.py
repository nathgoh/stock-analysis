import yaml
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
from kafka.serializer import DefaultSerializer

from .producer import produce_historical_news
from .stream import news_stream

TOPIC_NUM_PARTITIONS = 2
TOPIC_REPLICATION_FACTOR = 1


def ensure_topic(bootstrap_servers: str, topic: str) -> None:
    admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
    try:
        admin.create_topics(
            [
                NewTopic(
                    name=topic,
                    num_partitions=TOPIC_NUM_PARTITIONS,
                    replication_factor=TOPIC_REPLICATION_FACTOR,
                )
            ]
        )
    except TopicAlreadyExistsError:
        pass
    finally:
        admin.close()


def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open("configs/stocks.yaml", "r") as f:
        stocks_config = yaml.safe_load(f)

    ensure_topic(
        config["kafka"]["bootstrap_servers"], config["kafka"]["topics"]["live"]
    )
    ensure_topic(
        config["kafka"]["bootstrap_servers"], config["kafka"]["topics"]["historical"]
    )

    news_producer = KafkaProducer(
        bootstrap_servers=config["kafka"]["bootstrap_servers"],
        key_serializer=DefaultSerializer(),
        value_serializer=DefaultSerializer(),
        acks="all",
    )

    try:
        produce_historical_news(
            news_producer,
            config["kafka"]["topics"]["historical"],
            stocks_config["stocks"],
        )
        news_producer.flush()

        news_stream(
            news_producer, config["kafka"]["topics"]["live"], stocks_config["stocks"]
        )
    finally:
        news_producer.flush()
        news_producer.close()


if __name__ == "__main__":
    main()
