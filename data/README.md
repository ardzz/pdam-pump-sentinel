# Data Directory

Dataset SKAB tidak di-commit ke repo (lihat `.gitignore`). Download manual:

## SKAB (Skoltech Anomaly Benchmark)

- Repo: https://github.com/waico/SKAB
- License: GPL-3.0
- Kaggle DOI: 10.34740/KAGGLE/DSV/1693952

```bash
git clone https://github.com/waico/SKAB.git /tmp/skab
cp -r /tmp/skab/data ./data/skab
```

## Struktur yang Diharapkan

```text
data/skab/
├── anomaly-free/
│   └── anomaly-free.csv
├── valve1/
├── valve2/
└── other/
```

## Kolom Sensor

- `datetime` — timestamp
- `Accelerometer1RMS`, `Accelerometer2RMS` — vibration
- `Current` — motor current
- `Pressure` — hydraulic pressure
- `Temperature`, `Thermocouple` — thermal
- `Voltage` — supply
- `Volume Flow RateRMS` — flow
- `anomaly` — label (0/1)
- `changepoint` — changepoint label (0/1)
