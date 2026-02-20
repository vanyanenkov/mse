from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="AutoML YOLO",
    description="Автоматический подбор гиперпараметров YOLO",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#app.mount("/static", StaticFiles(directory="static"), name="static")
#app.mount("/runs", StaticFiles(directory="runs"), name="runs")

@app.get("/ping")
async def ping():

    return {"status": "ok"}

@app.get("/")
async def root():
    return {
        "message": "AutoML YOLO API is running",
        "docs": "/docs",
        "test_endpoint": "/ping"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)