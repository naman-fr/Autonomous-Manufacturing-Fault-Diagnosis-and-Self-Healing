from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from amfd.core.models import SensorWindow


class KafkaSensorConsumer:
    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str = "amfd-diagnosis",
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id

    def stream(self) -> Iterator[SensorWindow]:
        try:
            from confluent_kafka import Consumer
        except ImportError as exc:  # pragma: no cover - optional production dependency
            raise RuntimeError("Install confluent-kafka to enable Kafka streaming.") from exc

        consumer = Consumer(
            {
                "bootstrap.servers": self.bootstrap_servers,
                "group.id": self.group_id,
                "auto.offset.reset": "latest",
            }
        )
        consumer.subscribe([self.topic])
        try:
            while True:
                message = consumer.poll(1.0)
                if message is None or message.error():
                    continue
                value = message.value()
                if value is None:
                    continue
                payload: dict[str, Any] = json.loads(value.decode("utf-8"))
                yield SensorWindow(**payload)
        finally:
            consumer.close()
