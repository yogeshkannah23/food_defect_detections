# main.py
from fastapi import FastAPI
from routers import process

app = FastAPI()
app.include_router(process.router, prefix="/process")
