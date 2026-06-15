# Driverless Vehicle Edge-Case Intelligence Platform

Web platform for uploading dashcam videos, storing incident records, and reviewing ML analysis outputs (classification, risk timeline, semantic tags, summaries).

**Stack:** React + TypeScript (Vite) · FastAPI · PostgreSQL · AWS S3 · Redis

---

## Prerequisites

| Tool | Purpose |
|------|---------|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | PostgreSQL, Redis |
| Python 3.11+ | FastAPI backend |
| Node.js 20+ | React frontend |
| AWS account | S3 bucket + IAM credentials for video storage |

---

## First-time setup

Run these once after cloning the repo.

### 1. Start infrastructure (Docker)

From the project root:

```bash
docker compose up -d
```

This starts PostgreSQL (`5432`) and Redis (`6379`).

### 2. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `backend/.env` with your AWS and database settings:

```env
DATABASE_URL=postgresql://dev:dev@localhost:5432/edgecase
AUTH_SECRET=replace_with_a_long_random_string

AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
S3_BUCKET_NAME=your-bucket-name
AWS_REGION=eu-west-2

ML_API_URL=https://your-ml-api/v1/traffic-events/analyze-url?token=your_token
ML_API_VERIFY_SSL=0
ML_API_VIDEO_SOURCE=github
GITHUB_REPO=your-github-user/your-relay-repo
GITHUB_TOKEN=your_github_pat
GITHUB_MIRROR=https://gh-proxy.com/
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

Edit `frontend/.env` and set your [Mapbox public token](https://account.mapbox.com/):

```env
VITE_MAPBOX_TOKEN=pk.your_token_here
```

---

## Start the application (every day)

You need **three terminals**.

### Terminal 1 — Infrastructure

```bash
docker compose up -d
```

### Terminal 2 — API

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

On startup the API verifies the S3 bucket is reachable.

### Terminal 3 — Web app

```bash
cd frontend
npm run dev
```

---

## Main web app

http://localhost:5173

---

## Deploying

The simplest launch path is:

- Frontend: Vercel, Netlify, or another static host
- Backend: Render, Railway, Fly.io, or another Python web service host
- Database: managed PostgreSQL
- Redis: managed Redis/Valkey if you keep Redis-backed workers
- Video storage: AWS S3

Before deploying, rotate any secrets that have been used locally and store the new
values only in your hosting provider's environment variable settings.

### Backend

Deploy the `backend` directory as a Python web service.

Recommended commands:

```bash
pip install -r requirements.txt
```

```bash
sh start.sh
```

`start.sh` runs `alembic upgrade head` before starting FastAPI, so production
schema changes are applied before the API serves traffic.

Set these production environment variables:

```env
DATABASE_URL=postgresql://...
AUTH_SECRET=replace_with_a_long_random_string
CORS_ORIGINS=https://your-frontend-domain.vercel.app

AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
S3_BUCKET_NAME=your-bucket-name
AWS_REGION=eu-west-2

ML_API_URL=https://your-ml-api/v1/traffic-events/analyze-url?token=your_token
ML_API_VERIFY_SSL=0
ML_API_VIDEO_SOURCE=github
GITHUB_REPO=your-github-user/your-relay-repo
GITHUB_TOKEN=your_github_pat
GITHUB_MIRROR=https://gh-proxy.com/
ML_API_GITHUB_MIRROR=direct
```

Use the internal/private database URL when your backend and database are on the
same hosting provider.

### Frontend

Deploy the `frontend` directory as a Vite static app.

Recommended settings:

```bash
npm install
npm run build
```

Output directory:

```text
dist
```

Set these frontend environment variables:

```env
VITE_MAPBOX_TOKEN=pk.your_token_here
VITE_API_URL=https://your-backend-domain.onrender.com
```

Keep `VITE_API_URL` unset for local development; Vite will continue proxying
`/api` to `http://localhost:8000`.

### Production Smoke Test

After both services are deployed:

1. Visit the frontend URL.
2. Create an account.
3. Open an existing incident.
4. Upload a short video.
5. Confirm the video appears, ML analysis runs, and the incident detail page updates.

---

## Project layout

```
├── backend/          FastAPI app, SQLAlchemy models, Alembic migrations, S3 helpers
├── frontend/         React + TypeScript (Vite)
├── docker-compose.yml   PostgreSQL, Redis
```
