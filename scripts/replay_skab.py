import argparse
import json
import os
import sys
import time
from pathlib import Path

import paho.mqtt.client as mqtt

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.datasets.skab_loader import iter_telemetry_records, load_skab_csv  # noqa: E402


def build_telemetry_topic(station: str) -> str:
    return f'factory/skab/{station}/telemetry'


def build_telemetry_payload(record: dict, dry_run: bool = True) -> dict:
    payload = {
        'station': record['station'],
        'timestamp': record['timestamp'],
        'sensors': record['sensors'],
        'labels': record['labels'],
    }
    if dry_run:
        payload['dry_run'] = True
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description='Replay SKAB dataset to MQTT')
    parser.add_argument('--input', type=Path, required=True, help='Path to SKAB CSV file')
    parser.add_argument('--station', type=str, default='ipa_01', help='Station identifier')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of records to publish')
    parser.add_argument('--dry-run', action='store_true', help='Print payloads instead of publishing')
    parser.add_argument('--host', type=str, default=os.getenv('MQTT_HOST', 'localhost'), help='MQTT broker host')
    parser.add_argument('--port', type=int, default=int(os.getenv('MQTT_PORT', '1883')), help='MQTT broker port')
    parser.add_argument('--qos', type=int, default=int(os.getenv('MQTT_QOS', '1')), help='MQTT QoS level')
    args = parser.parse_args()

    frame = load_skab_csv(args.input)
    records = iter_telemetry_records(frame, station=args.station)

    if args.limit is not None:
        records = list(records)[:args.limit]

    client = None
    if not args.dry_run:
        client = mqtt.Client()
        client.connect(args.host, args.port)
        client.loop_start()

    try:
        for record in records:
            payload = build_telemetry_payload(record, dry_run=args.dry_run)
            topic = build_telemetry_topic(record['station'])
            if args.dry_run:
                print(f'Topic: {topic}')
                print(json.dumps(payload, indent=2))
            else:
                if client is not None:
                    client.publish(topic, json.dumps(payload), qos=args.qos)
            time.sleep(0.1)
    finally:
        if client is not None:
            client.loop_stop()
            client.disconnect()


if __name__ == '__main__':
    main()
