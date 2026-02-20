# boxctron-describes

A FastAPI microservice for generating descriptive information from images using AI vision models. Images can either be supplied as file URIs or as file uploads.

## Requirements

- Python 3.10 or higher
- Virtual environment (venv)

## Installation

### 1. Clone the repository

```bash
cd boxctron-describes
```

### 2. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example environment file and update with your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
- `AZURE_OPENAI_API_KEY` - Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT` - Your Azure OpenAI endpoint URL
- `GOOGLE_API_KEY` - (Optional) Google Gemini API key
- `ANTHROPIC_API_KEY` - (Optional) Anthropic Claude API key

### 5. Configure Authentication

The service supports hybrid authentication with both API keys and HTTP Basic authentication.

**Authentication Settings:**
- `AUTH_ENABLED` - Set to `true` to enable authentication, `false` to disable (default: `true`)
- `API_KEYS` - Comma-separated list of valid API keys for service-to-service communication
- `AUTH_USERNAME` - Username for HTTP Basic authentication (for browser access)
- `AUTH_PASSWORD` - Password for HTTP Basic authentication

**Generate secure API keys:**
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Example authentication configuration in `.env`:**
```bash
# Enable authentication
AUTH_ENABLED=true

# API keys for applications (comma-separated)
API_KEYS=AbCdEf123456_SecureKey1,XyZ789_SecureKey2

# HTTP Basic auth for browser access
AUTH_USERNAME=admin
AUTH_PASSWORD=your-secure-password
```

**Development/Testing:**
To disable authentication during development:
```bash
AUTH_ENABLED=false
```

## Running the Application

### Development Mode

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Interactive docs (Swagger UI): http://localhost:8000/docs
- Alternative docs (ReDoc): http://localhost:8000/redoc

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Health Check

```bash
GET /health
```

Returns the health status of the service.

### Describe Image

```bash
### Upload Endpoint

```
POST /api/v1/describe/upload
```

Process an uploaded image file and return descriptive information.

**Parameters:**
- `file` (required): Uploaded image file (multipart/form-data)
- `context` (optional): Additional context to guide description

The filename and MIME type are automatically extracted from the uploaded file.

**Example with cURL (API Key):**

```bash
curl -X POST "http://localhost:8000/api/v1/describe/upload" \
  -H "X-API-Key: your-api-key-here" \
  -F "file=@/path/to/image.jpg" \
  -F "context=Product photo"
```

**Example with cURL (Basic Auth):**

```bash
curl -X POST "http://localhost:8000/api/v1/describe/upload" \
  -u username:password \
  -F "file=@/path/to/image.jpg" \
  -F "context=Product photo"
```

### URI Endpoint

```
POST /api/v1/describe/uri
```

Process an image from a URI (file://, http://, or https://) and return descriptive information.

**Parameters:**
- `uri` (required): URI to the image file
- `filename` (required): Name of the file
- `mimetype` (required): MIME type of the image
- `context` (optional): Additional context to guide description

**Example with cURL (API Key):**

```bash
curl -X POST "http://localhost:8000/api/v1/describe/uri" \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "uri": "https://example.com/image.jpg",
    "filename": "image.jpg",
    "mimetype": "image/jpeg",
    "context": "Product photo"
  }'
```

**Example with cURL (Basic Auth):**

```bash
curl -X POST "http://localhost:8000/api/v1/describe/uri" \
  -u username:password \
  -H "Content-Type: application/json" \
  -d '{
    "uri": "https://example.com/image.jpg",
    "filename": "image.jpg",
    "mimetype": "image/jpeg",
    "context": "Product photo"
  }'
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_describe.py

# Run with verbose output
pytest -v
```

Test coverage report will be generated in `htmlcov/index.html`.

## Configuration

Configuration is managed through environment variables using `pydantic-settings`. Key settings include:

- **API Configuration**: App name, version, debug mode
- **Server Configuration**: Host, port
- **LLM Provider Keys**: Azure OpenAI, Google, Anthropic
- **LiteLLM Settings**: Model selection, temperature, max tokens
- **File Upload Settings**: Max size, allowed MIME types

See [.env.example](.env.example) for all available configuration options.

## Development

### Adding New Endpoints

1. Create a new router file in `app/api/routes/`
2. Define request/response models in `app/models/`
3. Register the router in `main.py`
4. Add tests in `tests/`

### Code Quality

The project uses:
- Type hints throughout the codebase
- Pydantic for data validation
- FastAPI's built-in OpenAPI documentation
- Pytest for testing

## Next Steps

The current implementation provides the basic project structure and API skeleton. To complete the service, you'll need to:

1. Implement the actual image processing logic in the describe endpoint
2. Add LiteLLM integration for calling vision models
3. Handle image downloading from URIs
4. Implement proper error handling and logging
5. Add authentication/authorization if needed
6. Configure CORS settings for production
7. Add Docker support for containerization
8. Set up CI/CD pipelines

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
