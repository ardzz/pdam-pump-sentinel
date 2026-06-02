from __future__ import annotations

from routemq.model import Model  # type: ignore[reportMissingImports]
from sqlalchemy import JSON, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column


class SensorReading(Model):
    __tablename__ = 'sensor_readings'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    station: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    source_timestamp: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sensors: Mapped[dict] = mapped_column(JSON, nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    anomaly: Mapped[int | None] = mapped_column(Integer, nullable=True)
