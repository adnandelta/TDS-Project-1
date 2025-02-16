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
import traceback
import json
from dotenv import load_dotenv
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
import os
import logging
from typing import Dict, Callable
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


load_dotenv()
AUTH_TOKEN = os.getenv("AIPROXY_TOKEN")
COMPLETION_ENDPOINT = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
VECTOR_ENDPOINT = "http://aiproxy.sanand.workers.dev/openai/v1/embeddings"

app = FastAPI()

RUNNING_IN_CODESPACES = "CODESPACES" in os.environ
RUNNING_IN_DOCKER = os.path.exists("/.dockerenv")
logging.basicConfig(level=logging.INFO)


def ensure_local_path(file_path: str) -> str:
    """Ensure the path uses './data/...' locally, but '/data/...' in Docker."""
    if (not RUNNING_IN_CODESPACES) and RUNNING_IN_DOCKER:
        print(
            "IN HERE", RUNNING_IN_DOCKER
        )  # If absolute Docker path, return as-is :  # If absolute Docker path, return as-is
        return file_path

    else:
        logging.info(f"Inside ensure_local_path with path: {file_path}")
        return file_path.lstrip("/")


operation_registry: Dict[str, Callable] = {
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


def parse_task_description(instruction_text: str, available_tools: list):
    api_response = requests.post(
        COMPLETION_ENDPOINT,
        headers={
            "Authorization": f"Bearer {AUTH_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "ou are a smart assistant capable of comprehending and analyzing tasks. You swiftly determine the most effective tool functions to achieve the desired outcomes.",
                },
                {"role": "user", "content": instruction_text},
            ],
            "tools": available_tools,
            "tool_choice": "required",
        },
    )
    return api_response.json()["choices"][0]["message"]


def execute_function_call(operation_call):
    logging.info(f"Inside execute_function_call with operation_call: {operation_call}")
    try:
        operation_name = operation_call["name"]
        operation_args = json.loads(operation_call["arguments"])
        operation_handler = operation_registry.get(operation_name)
        logging.info("PRINTING RESPONSE:::" * 3)
        print("Calling function:", operation_name)
        print("Arguments:", operation_args)
        if operation_handler:
            operation_handler(**operation_args)
        else:
            raise ValueError(f"Function {operation_name} not found")
    except Exception as e:
        error_details = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error executing function in execute_function_call: {str(e)}",
            headers={"X-Traceback": error_details},
        )


@app.post("/run")
async def run_task(
    instruction: str = Query(..., description="Plain-English task description")
):
    available_tools = [
        convert_function_to_openai_schema(func) for func in operation_registry.values()
    ]
    logging.info(len(available_tools))
    logging.info(f"Inside run_task with task: {instruction}")
    try:
        operation_response = parse_task_description(
            instruction, available_tools
        )  # returns  message from response
        if operation_response["tool_calls"]:
            for operation in operation_response["tool_calls"]:
                execute_function_call(operation["function"])
        return {"status": "success", "message": "Task executed successfully"}
    except Exception as e:
        error_details = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error executing function in run_task: {str(e)}",
            headers={"X-Traceback": error_details},
        )


@app.get("/read", response_class=PlainTextResponse)
async def read_file(file_path: str = Query(..., description="Path to the file to read")):
    logging.info(f"Inside read_file with path: {file_path}")
    target_path = ensure_local_path(file_path)
    if not os.path.exists(target_path):
        raise HTTPException(
            status_code=500, detail=f"Error executing function in read_file (GET API"
        )
    with open(target_path, "r") as file:
        file_content = file.read()
    return file_content


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)