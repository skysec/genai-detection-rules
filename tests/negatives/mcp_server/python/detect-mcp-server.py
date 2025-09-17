# NEGATIVE TEST CASES - Should NOT be detected by the MCP server rule

# Regular Python imports - should NOT match
import json
import asyncio
import httpx
import os
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel

# Regular class definitions - should NOT match
class DataProcessor:
    def __init__(self):
        self.data = []

    def process(self, input_data):
        return input_data.upper()

class WebServer:
    def __init__(self, port):
        self.port = port

    def start(self):
        print(f"Starting server on port {self.port}")

class APIHandler:
    def handle_request(self, request):
        return {"status": "success"}

# Regular function definitions - should NOT match
def process_data(data):
    return data * 2

async def fetch_data(url):
    return {"data": "example"}

def handle_http_request(request):
    return {"status": 200, "body": "OK"}

def calculate_result(x, y):
    return x + y

# Regular decorators - should NOT match
@dataclass
class User:
    name: str
    email: str
    age: int

@property
def full_name(self):
    return f"{self.first_name} {self.last_name}"

from functools import wraps

@wraps
def logged(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@logged
def some_function():
    return "result"

# Regular async patterns - should NOT match
async def main():
    data = await fetch_api_data()
    return process_results(data)

async def handle_request(request):
    return {"processed": request}

# Flask/FastAPI patterns - should NOT match
from flask import Flask
from fastapi import FastAPI

app = Flask(__name__)
api = FastAPI()

@app.route("/api/data")
def get_data():
    return {"data": "value"}

@api.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

# Regular server operations - should NOT match
server = WebServer(8080)
server.start()

http_server.run()
app.run()
api.run()

# Regular tool/utility functions - should NOT match
def tool_helper(data):
    return data.strip()

def resource_loader(path):
    with open(path) as f:
        return f.read()

def prompt_formatter(text):
    return f"Prompt: {text}"

# Regular context usage - should NOT match
def function_with_context(data, context=None):
    if context:
        print(f"Context: {context}")
    return data

class Context:
    def __init__(self):
        self.data = {}

# Regular run patterns - should NOT match
def run_process():
    print("Running process")

process.run()
application.run()
script.run()

if __name__ == "__main__":
    run_process()

if __name__ == "__main__":
    app = create_app()
    app.run()

# Database patterns - should NOT match
from sqlalchemy import create_engine
from django.db import models

class DatabaseModel(models.Model):
    name = models.CharField(max_length=100)

# Machine learning patterns that are not MCP - should NOT match
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
import torch

model = RandomForestClassifier()
neural_net = keras.Sequential()
tensor = torch.tensor([1, 2, 3])

# API client patterns - should NOT match
import requests
import aiohttp

response = requests.get("https://api.example.com/data")
async with aiohttp.ClientSession() as session:
    async with session.get("https://api.example.com") as resp:
        data = await resp.json()