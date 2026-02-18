"""
File handling utilities for streaming uploads and downloads.
"""
import tempfile
import logging
from pathlib import Path
from typing import Optional, BinaryIO, AsyncIterator
from fastapi import UploadFile, HTTPException, status
import httpx

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1MB chunks


def _cleanup_temp_file(temp_file: BinaryIO, temp_path: Path) -> None:
    """
    Clean up a temporary file by closing and deleting it.

    Args:
        temp_file: The open file object to close
        temp_path: Path to the temporary file to delete
    """
    temp_file.close()
    if temp_path.exists():
        temp_path.unlink()


async def _write_chunks_to_file(
    temp_file: BinaryIO,
    temp_path: Path,
    chunks: AsyncIterator[bytes],
    max_size: int
) -> None:
    """
    Write chunks from an async iterator to a file, enforcing size limits.

    Args:
        temp_file: The open file object to write to
        temp_path: Path to the temporary file (for cleanup on error)
        chunks: Async iterator yielding byte chunks
        max_size: Maximum allowed total file size in bytes

    Raises:
        HTTPException: If total size exceeds max_size
    """
    total_size = 0
    async for chunk in chunks:
        total_size += len(chunk)
        if total_size > max_size:
            _cleanup_temp_file(temp_file, temp_path)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum allowed size of {max_size} bytes"
            )
        temp_file.write(chunk)


async def _read_upload_chunks(file: UploadFile) -> AsyncIterator[bytes]:
    """
    Read chunks from an uploaded file.

    Args:
        file: The uploaded file to read from

    Yields:
        Byte chunks of size CHUNK_SIZE
    """
    while chunk := await file.read(CHUNK_SIZE):
        yield chunk


async def stream_upload_to_temp(
    file: UploadFile,
    filename: str,
    max_size: int
) -> Path:
    """
    Stream an uploaded file to a temporary file while enforcing size limits.

    This avoids loading large files entirely into memory by streaming in chunks.

    Args:
        file: The uploaded file to stream
        filename: Original filename (used for file extension)
        max_size: Maximum allowed file size in bytes

    Returns:
        Path to the temporary file

    Raises:
        HTTPException: If file size exceeds limit or streaming fails
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix)
    temp_path = Path(temp_file.name)

    try:
        await _write_chunks_to_file(
            temp_file,
            temp_path,
            _read_upload_chunks(file),
            max_size
        )
        temp_file.close()
        return temp_path

    except HTTPException:
        raise
    except Exception as e:
        _cleanup_temp_file(temp_file, temp_path)
        logger.error(f"Error streaming uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing uploaded file"
        )


async def stream_url_to_temp(
    url: str,
    max_size: int,
    filename: Optional[str] = None,
    timeout: float = 30.0
) -> Path:
    """
    Stream a file from a URL to a temporary file while enforcing size limits.

    This avoids loading large files entirely into memory by streaming in chunks.

    Args:
        url: The URL to download from
        max_size: Maximum allowed file size in bytes
        filename: Optional filename for file extension (inferred from URL if not provided)
        timeout: Request timeout in seconds

    Returns:
        Path to the temporary file

    Raises:
        HTTPException: If download fails, file size exceeds limit, or streaming fails
    """
    # Determine file suffix
    if filename:
        suffix = Path(filename).suffix
    else:
        suffix = Path(url).suffix or ""

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp_file.name)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    _cleanup_temp_file(temp_file, temp_path)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to download file from URL: HTTP {response.status_code}"
                    )

                await _write_chunks_to_file(
                    temp_file,
                    temp_path,
                    response.aiter_bytes(chunk_size=CHUNK_SIZE),
                    max_size
                )

        temp_file.close()
        return temp_path

    except HTTPException:
        raise
    except httpx.RequestError as e:
        _cleanup_temp_file(temp_file, temp_path)
        logger.error(f"Error downloading file from URL {url}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download file from URL: {str(e)}"
        )
    except Exception as e:
        _cleanup_temp_file(temp_file, temp_path)
        logger.error(f"Error streaming file from URL {url}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing file from URL"
        )
