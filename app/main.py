# /// script
# dependencies = [
#   "fastapi",
#   "requests",
#   "python-dotenv",
#   "uvicorn",
#   "python-dotenv",
#   "beautifulsoup4",
#   "markdown",
#   "requests<3",
#   "duckdb",
#   "numpy",
#   "python-dateutil",
#   "docstring-parser",
#   "httpx",
#   "pydantic",
# ]
# ///
from typing import Any, Dict, Callable, Optional
import os
import json
import logging
import traceback
from pathlib import Path
from contextlib import contextmanager

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import PlainTextResponse

from funtion_tasks import (
    format_file_with_prettier,
    convert_function_to_openai_schema,
    query_gpt,
    query_gpt_image,
    query_database,
    extract_specific_text_using_llm,
    get_embeddings,
    get_similar_text_using_embeddings,
    extract_text_from_image,
    extract_specific_content_and_create_index,
    process_and_write_logfiles,
    sort_json_by_keys,
    count_occurrences,
    install_and_run_script,
)

class ConfigurationManager:
    def __init__(self):
        load_dotenv()
        self.api_token = os.getenv("AIPROXY_TOKEN")
        self.api_endpoints = {
            "chat": "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            "embedding": "http://aiproxy.sanand.workers.dev/openai/v1/embeddings"
        }
        self.environment = {
            "is_codespaces": "CODESPACES" in os.environ,
            "is_docker": os.path.exists("/.dockerenv")
        }

class TaskProcessor:
    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.available_functions = {
            "install_and_run_script": install_and_run_script,
            "format_file_with_prettier": format_file_with_prettier,
            "query_database": query_database,
            "extract_specific_text_using_llm": extract_specific_text_using_llm,
            "get_similar_text_using_embeddings": get_similar_text_using_embeddings,
            "extract_text_from_image": extract_text_from_image,
            "extract_specific_content_and_create_index": extract_specific_content_and_create_index,
            "process_and_write_logfiles": process_and_write_logfiles,
            "sort_json_by_keys": sort_json_by_keys,
            "count_occurrences": count_occurrences,
        }

    def normalize_path(self, file_path: str) -> str:
        if not self.config.environment["is_codespaces"] and self.config.environment["is_docker"]:
            return file_path
        return file_path.lstrip("/")

    def analyze_task(self, task_description: str, available_tools: list) -> dict:
        response = requests.post(
            self.config.api_endpoints["chat"],
            headers={
                "Authorization": f"Bearer {self.config.api_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an advanced task analyzer that identifies optimal tool functions for achieving desired outcomes.",
                    },
                    {"role": "user", "content": task_description},
                ],
                "tools": available_tools,
                "tool_choice": "required",
            },
        )
        return response.json()["choices"][0]["message"]

    def execute_tool(self, tool_call: dict) -> None:
        try:
            tool_name = tool_call["name"]
            tool_args = json.loads(tool_call["arguments"])
            
            if tool_func := self.available_functions.get(tool_name):
                logging.info(f"Executing tool: {tool_name}")
                logging.info(f"Arguments: {tool_args}")
                tool_func(**tool_args)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as e:
            error_trace = traceback.format_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Tool execution failed: {str(e)}",
                headers={"X-Error-Trace": error_trace},
            )

class APIServer:
    def __init__(self):
        self.config = ConfigurationManager()
        self.processor = TaskProcessor(self.config)
        self.app = FastAPI()
        self._setup_routes()
        logging.basicConfig(level=logging.INFO)

    def _setup_routes(self):
        @self.app.post("/run")
        async def handle_task(task: str = Query(..., description="Natural language task description")):
            try:
                tools = [convert_function_to_openai_schema(func) 
                        for func in self.processor.available_functions.values()]
                
                task_analysis = self.processor.analyze_task(task, tools)
                
                if task_analysis.get("tool_calls"):
                    for tool in task_analysis["tool_calls"]:
                        self.processor.execute_tool(tool["function"])
                
                return {"status": "success", "message": "Task completed successfully"}
            except Exception as e:
                error_trace = traceback.format_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Task execution failed: {str(e)}",
                    headers={"X-Error-Trace": error_trace},
                )

        @self.app.get("/read", response_class=PlainTextResponse)
        async def read_content(path: str = Query(..., description="Target file path")):
            try:
                file_path = self.processor.normalize_path(path)
                if not Path(file_path).exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                    
                return Path(file_path).read_text()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"File reading failed: {str(e)}")

    def start(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    server = APIServer()
    server.start()
