# Crochet Pattern Converter API

FastAPI backend for the crochet pattern converter. This milestone validates and temporarily stores an uploaded PDF, then returns a fixed mock conversion. It does **not** render the PDF, run YOLO, or interpret the uploaded chart yet.

## Setup

Python 3.11 or newer is recommended.

### Windows PowerShell

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Configuration

The API reads these optional environment variables:

- `MAX_UPLOAD_SIZE_BYTES`: maximum accepted upload size; defaults to `10485760` (10 MB).
- `FRONTEND_ORIGINS`: comma-separated CORS allowlist; defaults to `http://localhost:3000,http://127.0.0.1:3000`.

The project intentionally does not add a dotenv loader. Copy values from `.env.example` into your shell or process manager when overriding the defaults.

For the frontend, copy `frontend/.env.example` to `frontend/.env.local`. Its `NEXT_PUBLIC_API_URL` should point to this server:

```text
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Run the API

From `backend/` with the virtual environment active:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Available routes:

- `GET /api/health`
- `POST /api/convert`, with a multipart PDF field named `file`
- Interactive API documentation at `http://localhost:8000/docs`

## Run tests

From `backend/`:

```bash
python -m pytest
```

The tests use an in-process FastAPI test client and require no model, server, network connection, or external service.

## Future YOLO integration

`app/services/detectors/base.py` defines the detector boundary. A future `YoloSymbolDetector` should implement `SymbolDetector.detect(RenderedPage)` and return normalized detections containing the symbol ID, class, confidence, page number, and rotated bounding box.

PDF rasterization and chart-to-instruction interpretation belong in `ConversionService`. Register the YOLO implementation in `create_app()` in place of `MockSymbolDetector`; the route and frontend response contract can remain unchanged.
