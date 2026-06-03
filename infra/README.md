# Infrastructure notes

- Prometheus scrapes the RouteMQ application at `app:8080/metrics`, matching RouteMQ's default metrics path and port. The current `bootstrap/app.py` does not start RouteMQ's `HealthServer` metrics renderer, so live metrics exposure remains deferred until the app entrypoint wires the env-gated metrics server safely.
