# Project Context for Qwen Code

## Project Overview

This project is a **JetBrains AI to OpenAI API adapter**. It provides a high-performance, asynchronous proxy service that translates requests for JetBrains AI's large language models into the OpenAI API format. This allows existing tools and clients that use the OpenAI API to seamlessly interact with JetBrains AI models.

The service is built using Python with `FastAPI` and `httpx`, designed for high concurrency and supports true streaming responses. It features intelligent account quota management by automatically rotating JetBrains accounts that have exceeded their usage limits. The adapter also supports both OpenAI and Anthropic SDKs, including Function Calling (Tools) and stream/non-stream responses for both.

### Core Features

- **High-Concurrency Async Architecture**: Built with `httpx` and `FastAPI` to handle thousands of concurrent connections efficiently.
- **Full OpenAI Compatibility**: Integrates with existing OpenAI clients and tools without modification.
- **Anthropic API Compatibility**: Supports the `/v1/messages` endpoint for Anthropic SDKs.
- **Dynamic Authentication**: Supports automatic JWT refresh and pooling for JetBrains accounts.
- **Intelligent Quota Management**: Automatically detects and rotates accounts that exceed their quota.
- **Function Calling Support**: Fully implements OpenAI's `tools` and `tool_calls` functionalities.
- **Easy Deployment**: Offers Docker-based deployment for quick setup.

## Project Structure

```
jetbrainsai2api/
├── main.py              # Main application (async server + API adapter)
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker build file
├── docker-compose.yml  # Docker Compose configuration
├── jetbrainsai.json     # JetBrains AI JWT/account configuration
├── client_api_keys.json # Client API keys for accessing this proxy
└── models.json         # Available model list and Anthropic model mappings
```

## Key Configuration Files

1.  **`jetbrainsai.json`**: Stores the JetBrains AI account credentials. It supports two formats:
    -   **Automatic Refresh (Recommended)**: Using `licenseId` and `authorization` for automatic JWT refresh.
    -   **Static JWT**: Using a pre-obtained JWT directly.
    -   The application will automatically rotate between these accounts and manage their quotas.

2.  **`client_api_keys.json`**: Contains a list of API keys that clients must use to authenticate with this proxy service.

3.  **`models.json`**: Defines the list of models available from JetBrains AI that this proxy can serve. It also includes a mapping section (`anthropic_model_mappings`) to translate Anthropic model names to the corresponding JetBrains model identifiers.

## Building and Running

### Prerequisites

- Python 3.11+
- `pip` for dependency management
- Docker & Docker Compose (for containerized deployment)

### Local Development

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure Secrets**:
    - Create `jetbrainsai.json`, `client_api_keys.json`, and `models.json` based on the examples in the README or the default files created on first run.
3.  **Run the Server**:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

### Docker Deployment (Recommended)

1.  **Configure Secrets**: Ensure `jetbrainsai.json`, `client_api_keys.json`, and `models.json` exist in the project directory.
2.  **Start the Service**:
    ```bash
    docker-compose up -d
    ```
    This will build the image (if necessary) and start the service, mapping port 8000 on the host to the container.

## API Endpoints

- **GET `/v1/models`**: Lists the models available through this proxy. Requires a valid client API key.
- **POST `/v1/chat/completions`**: The standard OpenAI chat completion endpoint. Supports streaming and non-streaming responses, as well as function calling. Requires a valid client API key in the `Authorization: Bearer <key>` header.
- **POST `/v1/messages`**: The Anthropic-compatible messages endpoint. Supports streaming and non-streaming responses. Requires a valid client API key in the `x-api-key: <key>` header and the `x-anthropic-version` header.

## Development Conventions

- The codebase is written in Python using `FastAPI`.
- Asynchronous programming (`async`/`await`) is used extensively for handling I/O-bound operations like HTTP requests.
- Pydantic models are used for request and response data validation.
- Configuration is loaded from JSON files at startup.
- The code handles authentication token refresh and quota checking automatically.
- Logging is primarily done via `print` statements for simplicity, though a more robust logging framework could be integrated.
