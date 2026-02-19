"""Tests for file_handler utility functions."""
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock
import pytest
import httpx
import respx
from fastapi import HTTPException, UploadFile

from app.utils.file_handler import (
    stream_upload_to_temp,
    stream_url_to_temp,
    get_path_from_uri,
    CHUNK_SIZE
)


class TestStreamUploadToTemp:
    """Tests for stream_upload_to_temp function."""

    @pytest.mark.asyncio
    async def test_successful_upload(self):
        """Test successfully streaming an uploaded file to temp location."""
        # Create mock uploaded file
        test_content = b"Test image content"
        mock_file = Mock(spec=UploadFile)
        mock_file.read = AsyncMock(side_effect=[test_content, b""])  # Returns content then empty

        filename = "test_image.jpg"
        max_size = 1024 * 1024  # 1MB

        # Stream to temp file
        result_path = await stream_upload_to_temp(mock_file, filename, max_size)

        try:
            # Verify result
            assert result_path.exists()
            assert result_path.suffix == ".jpg"
            assert result_path.read_bytes() == test_content
        finally:
            # Clean up
            if result_path.exists():
                result_path.unlink()

    @pytest.mark.asyncio
    async def test_upload_exceeds_max_size(self):
        """Test that upload is rejected when it exceeds max size."""
        # Create mock file that's too large
        large_chunk = b"x" * (CHUNK_SIZE + 1)
        mock_file = Mock(spec=UploadFile)
        mock_file.read = AsyncMock(side_effect=[large_chunk, b""])

        filename = "large_image.jpg"
        max_size = CHUNK_SIZE  # Set limit to exactly one chunk

        # Should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await stream_upload_to_temp(mock_file, filename, max_size)

        assert exc_info.value.status_code == 413
        assert "exceeds maximum allowed size" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_upload_with_multiple_chunks(self):
        """Test streaming a file that comes in multiple chunks."""
        # Create mock file with multiple chunks
        chunk1 = b"First chunk of data"
        chunk2 = b"Second chunk of data"
        mock_file = Mock(spec=UploadFile)
        mock_file.read = AsyncMock(side_effect=[chunk1, chunk2, b""])

        filename = "chunked_image.png"
        max_size = 1024 * 1024

        result_path = await stream_upload_to_temp(mock_file, filename, max_size)

        try:
            assert result_path.exists()
            assert result_path.read_bytes() == chunk1 + chunk2
        finally:
            # Clean up
            if result_path.exists():
                result_path.unlink()

    @pytest.mark.asyncio
    async def test_upload_error_handling(self):
        """Test that errors during upload are handled properly."""
        # Mock file that raises an error
        mock_file = Mock(spec=UploadFile)
        mock_file.read = AsyncMock(side_effect=IOError("Read error"))

        filename = "error_image.jpg"
        max_size = 1024

        with pytest.raises(HTTPException) as exc_info:
            await stream_upload_to_temp(mock_file, filename, max_size)

        assert exc_info.value.status_code == 500


class TestStreamUrlToTemp:
    """Tests for stream_url_to_temp function."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_successful_download(self):
        """Test successfully downloading a file from URL."""
        test_content = b"Downloaded image content"

        # Mock the HTTP GET request
        respx.get("https://example.com/image.jpg").mock(
            return_value=httpx.Response(200, content=test_content)
        )

        result_path = await stream_url_to_temp(
            url="https://example.com/image.jpg",
            max_size=1024 * 1024,
            filename="downloaded_image.jpg"
        )

        try:
            assert result_path.exists()
            assert result_path.suffix == ".jpg"
            assert result_path.read_bytes() == test_content
        finally:
            # Clean up
            if result_path.exists():
                result_path.unlink()

    @pytest.mark.asyncio
    @respx.mock
    async def test_download_http_error(self):
        """Test handling of HTTP error during download."""
        # Mock HTTP 404 response
        respx.get("https://example.com/missing.jpg").mock(
            return_value=httpx.Response(404)
        )

        with pytest.raises(HTTPException) as exc_info:
            await stream_url_to_temp(
                url="https://example.com/missing.jpg",
                max_size=1024,
                filename="missing.jpg"
            )

        assert exc_info.value.status_code == 400
        assert "HTTP 404" in exc_info.value.detail

    @pytest.mark.asyncio
    @respx.mock
    async def test_download_exceeds_max_size(self):
        """Test that download is rejected when it exceeds max size."""
        large_content = b"x" * (CHUNK_SIZE + 1)

        # Mock response with large content
        respx.get("https://example.com/large.jpg").mock(
            return_value=httpx.Response(200, content=large_content)
        )

        with pytest.raises(HTTPException) as exc_info:
            await stream_url_to_temp(
                url="https://example.com/large.jpg",
                max_size=CHUNK_SIZE,
                filename="large.jpg"
            )

        assert exc_info.value.status_code == 413

    @pytest.mark.asyncio
    @respx.mock
    async def test_download_network_error(self):
        """Test handling of network error during download."""
        # Mock network error
        respx.get("https://example.com/image.jpg").mock(
            side_effect=httpx.RequestError("Connection failed")
        )

        with pytest.raises(HTTPException) as exc_info:
            await stream_url_to_temp(
                url="https://example.com/image.jpg",
                max_size=1024,
                filename="image.jpg"
            )

        assert exc_info.value.status_code == 400
        assert "Failed to download" in exc_info.value.detail


class TestGetPathFromUri:
    """Tests for get_path_from_uri function."""

    @pytest.mark.asyncio
    async def test_file_uri_existing_file(self):
        """Test converting file:// URI to path for existing file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(b"Test content")

        try:
            # Test with file:// URI
            file_uri = tmp_path.as_uri()
            result_path = await get_path_from_uri(
                uri=file_uri,
                max_size=1024,
                filename="test.jpg"
            )

            assert result_path == tmp_path
            assert result_path.exists()
        finally:
            # Clean up
            tmp_path.unlink()

    @pytest.mark.asyncio
    async def test_file_uri_missing_file(self):
        """Test that file:// URI for non-existent file raises error."""
        non_existent_path = Path("/tmp/non_existent_file_12345.jpg")
        file_uri = non_existent_path.as_uri()

        with pytest.raises(HTTPException) as exc_info:
            await get_path_from_uri(
                uri=file_uri,
                max_size=1024,
                filename="test.jpg"
            )

        assert exc_info.value.status_code == 400
        assert "File not found" in exc_info.value.detail

    @pytest.mark.asyncio
    @respx.mock
    async def test_http_uri_download(self):
        """Test that http:// URI triggers download."""
        test_content = b"HTTP downloaded content"

        # Mock the HTTP GET request
        respx.get("http://example.com/image.jpg").mock(
            return_value=httpx.Response(200, content=test_content)
        )

        result_path = await get_path_from_uri(
            uri="http://example.com/image.jpg",
            max_size=1024 * 1024,
            filename="downloaded.jpg"
        )

        try:
            assert result_path.exists()
            assert result_path.read_bytes() == test_content
        finally:
            # Clean up
            if result_path.exists():
                result_path.unlink()

    @pytest.mark.asyncio
    @respx.mock
    async def test_https_uri_download(self):
        """Test that https:// URI triggers download."""
        test_content = b"HTTPS downloaded content"

        # Mock the HTTPS GET request
        respx.get("https://example.com/secure-image.jpg").mock(
            return_value=httpx.Response(200, content=test_content)
        )

        result_path = await get_path_from_uri(
            uri="https://example.com/secure-image.jpg",
            max_size=1024 * 1024,
            filename="secure.jpg"
        )

        try:
            assert result_path.exists()
            assert result_path.read_bytes() == test_content
        finally:
            # Clean up
            if result_path.exists():
                result_path.unlink()

    @pytest.mark.asyncio
    async def test_unsupported_scheme(self):
        """Test that unsupported URI scheme raises error."""
        with pytest.raises(HTTPException) as exc_info:
            await get_path_from_uri(
                uri="ftp://example.com/file.jpg",
                max_size=1024,
                filename="test.jpg"
            )

        assert exc_info.value.status_code == 400
        assert "Unsupported URI scheme" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_file_uri_with_netloc(self):
        """Test file:// URI with network location (UNC path)."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(b"Test content")

        try:
            # Construct a file URI with netloc (simulating UNC path)
            # Note: This is system-dependent; on Unix it would be file://localhost/path
            file_uri = f"file://localhost{tmp_path}"

            result_path = await get_path_from_uri(
                uri=file_uri,
                max_size=1024,
                filename="test.jpg"
            )

            # The result should have the netloc incorporated
            assert result_path.exists()
        finally:
            # Clean up
            tmp_path.unlink()
