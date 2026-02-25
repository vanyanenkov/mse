from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.api import endpoints


app = FastAPI(
    title="AutoML YOLO API",
    description="API для автоматического подбора гиперпараметров YOLO",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path("static").mkdir(exist_ok=True)
Path("runs").mkdir(exist_ok=True)
Path("runs/detect").mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/runs", StaticFiles(directory="runs"), name="runs")

app.include_router(endpoints.router, prefix="/api")

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {
        "message": "AutoML YOLO API is running",
        "docs": "/docs",
        "test_endpoint": "/ping",
        "api_endpoints": {
            "start_automl": "/api/start",
            "get_status": "/api/status",
            "test_configs": "/api/configs"
        },
        "static_files": {
            "static": "/static/",
            "runs": "/runs/"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
