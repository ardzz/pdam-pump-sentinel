from routemq.router import Router  # type: ignore[reportMissingImports]

from app.controllers.anomaly_controller import Controller
from app.controllers.label_controller import LabelController
from app.middleware import (
    CorrelationLoggingMiddleware,
    InMemoryRateLimitMiddleware,
    ValidateTelemetryMiddleware,
)

router = Router()
router.on(
    'factory/skab/{station}/telemetry',
    Controller.ingest,
    qos=1,
    middleware=[
        ValidateTelemetryMiddleware(),
        InMemoryRateLimitMiddleware(),
        CorrelationLoggingMiddleware(),
    ],
)
router.on(
    'factory/skab/{station}/label',
    LabelController.ingest,
    qos=1,
    middleware=[
        InMemoryRateLimitMiddleware(),
        CorrelationLoggingMiddleware(),
    ],
)
