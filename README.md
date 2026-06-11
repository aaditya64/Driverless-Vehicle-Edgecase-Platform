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

## Project layout

```
├── backend/          FastAPI app, SQLAlchemy models, Alembic migrations, S3 helpers
├── frontend/         React + TypeScript (Vite)
├── docker-compose.yml   PostgreSQL, Redis
```
