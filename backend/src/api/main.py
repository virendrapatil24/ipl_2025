from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import chat

app = FastAPI(
    title="IPL Fantasy Predictor API",
    description="API for IPL Fantasy Predictor using RAG and ML",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api", tags=["chat"])


@app.get("/")
async def root():
    return {"message": "Welcome to IPL Fantasy Predictor API"}
