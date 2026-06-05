# Driverless Vehicle Edge-Case Intelligence Platform

Web platform for uploading dashcam videos, storing incident records, and reviewing ML analysis outputs (classification, risk timeline, semantic tags, summaries).

**Stack:** React + TypeScript (Vite) · FastAPI · PostgreSQL · MinIO (S3-compatible) · Redis

---

## Prerequisites

| Tool | Purpose |
|------|---------|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | PostgreSQL, Redis, MinIO |
| Python 3.11+ | FastAPI backend |
| Node.js 20+ | React frontend |

---

## First-time setup

Run these once after cloning the repo.

### 1. Start infrastructure (Docker)

From the project root:

```bash
docker compose up -d
```

This starts:

| Service | Port(s) | Credentials |
|---------|---------|-------------|
| PostgreSQL (PostGIS) | `5432` | user `dev`, password `dev`, database `edgecase` |
| Redis | `6379` | (no auth in dev) |
| MinIO (S3-compatible storage) | `9000` (API), `9001` (console) | `dev_access_key` / `dev_secret_key` |

Check containers are running:

```bash
docker compose ps
```

### 2. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `backend/.env`:

```env
DATABASE_URL=postgresql://dev:dev@localhost:5432/edgecase

S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=dev_access_key
AWS_SECRET_ACCESS_KEY=dev_secret_key
S3_BUCKET_NAME=edgecase-videos
AWS_REGION=eu-west-2
```

Apply database migrations:

```bash
alembic upgrade head
```

### 3. Frontend

```bash
cd frontend
npm install
cp .env.example .env
```

Edit `frontend/.env` and set your [Mapbox public token](https://account.mapbox.com/) (required for the **Map** page and upload location pin):

```env
VITE_MAPBOX_TOKEN=pk.your_token_here
```

---

## Start the application (every day)

You need **three terminals**. Order matters: Docker first, then API, then UI.

### Terminal 1 — Infrastructure

```bash
# from project root
docker compose up -d
```

### Terminal 2 — API

```bash
cd backend
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

On startup the API creates the MinIO bucket `edgecase-videos` if it does not exist.

### Terminal 3 — Web app

```bash
cd frontend
npm run dev
```

---

## URLs

| URL | What it is |
|-----|------------|
| http://localhost:5173 | **Main web app** (incident list, map, upload, detail) |
| http://localhost:5173/map | Interactive incident map (Mapbox) |
| http://localhost:8000/docs | FastAPI Swagger UI — test API endpoints |
| http://localhost:8000/health | API health check (`{"status":"ok"}`) |
| http://localhost:8000/incidents | Incident list (JSON) |
| http://localhost:9001 | **MinIO console** — browse uploaded videos |
| http://localhost:9000 | MinIO S3 API (not a browser UI) |

The frontend proxies `/api/*` to the backend (see `frontend/vite.config.ts`), so the UI talks to `http://localhost:5173/api/...` which forwards to port `8000`.

---

## Quick smoke test

1. Open http://localhost:5173
2. Go to **Upload** and submit a video (e.g. `dummy_dashcam.mp4` in the repo root)
3. You should land on the incident detail page with the video player
4. Confirm the file in MinIO: http://localhost:9001 → bucket `edgecase-videos` → `uploads/`
5. Confirm the DB row:

```bash
docker compose exec db psql -U dev -d edgecase -c \
  "SELECT id, status, uploaded_at FROM incidents ORDER BY uploaded_at DESC LIMIT 5;"
```

New uploads are stored with status **`waiting`**. ML processing (Celery worker) is not wired yet, so status will not advance to `completed` on its own.

---

## Stop everything

```bash
# Stop API and frontend: Ctrl+C in those terminals

# Stop Docker services (from project root)
docker compose down
```

Data persists in Docker volumes (`postgres_data`, `minio_data`) until you remove them:

```bash
docker compose down -v   # wipes DB and stored videos
```

---

## Project layout

```
├── backend/          FastAPI app, SQLAlchemy models, Alembic migrations, S3 helpers
├── frontend/         React + TypeScript (Vite)
├── docker-compose.yml   PostgreSQL, Redis, MinIO
└── dummy_dashcam.mp4    Sample video for testing uploads
```

---

## Troubleshooting

**Frontend shows “Could not load incidents”**  
→ Is `uvicorn` running on port 8000? Check http://localhost:8000/health

**Upload fails**  
→ Is Docker up? MinIO must be reachable at `localhost:9000`. Check `docker compose ps`.

**Database errors on API start**  
→ Run `docker compose up -d` and `alembic upgrade head` from `backend/`. Ensure `backend/.env` has `DATABASE_URL`.

**Video does not play on detail page**  
→ Presigned URLs point at MinIO (`localhost:9000`). MinIO must be running; open the incident again to refresh the URL (expires after ~1 hour).

**Port already in use**  
→ Another process may be bound to `5432`, `8000`, or `5173`. Stop the conflicting service or change the port in the relevant config.

---

## What is not running yet

- **Celery worker** — async ML pipeline (BADAS + Qwen3-VL); Redis is provisioned but unused
- **Delete incident API** — remove records manually via `psql` or MinIO console if needed
