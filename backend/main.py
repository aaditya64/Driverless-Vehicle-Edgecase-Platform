from fastapi import FastAPI

app = FastAPI(title="Driverless-Vehicle Edge-Case Platform")

@app.get("/health")
def health():
    return {"status": "ok"}