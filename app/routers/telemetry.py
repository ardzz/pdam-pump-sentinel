from routemq.router import Router  # type: ignore[reportMissingImports]

from app.controllers.anomaly_controller import Controller

router = Router()
router.on('factory/skab/{station}/telemetry', Controller.ingest, qos=1)
