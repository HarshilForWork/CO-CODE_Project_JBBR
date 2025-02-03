from fastapi import FastAPI
from backend.routes import router

app = FastAPI(title="MCQ Generator API")

# Include routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)