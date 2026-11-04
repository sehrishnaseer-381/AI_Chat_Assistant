import asyncio
import io

import pytest
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import Headers, UploadFile

from app.services.vision_service import (
    MAX_IMAGE_COUNT,
    MAX_IMAGE_SIZE_BYTES,
    validate_image_uploads,
)


def _image_bytes(image_format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color=(30, 80, 140)).save(
        buffer,
        format=image_format,
    )
    return buffer.getvalue()


def _upload(filename: str, content_type: str, content: bytes) -> UploadFile:
    return UploadFile(
        file=io.BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


@pytest.mark.parametrize(
    ("filename", "content_type", "image_format"),
    [
        ("photo.png", "image/png", "PNG"),
        ("photo.jpg", "image/jpeg", "JPEG"),
        ("photo.jpeg", "image/jpeg", "JPEG"),
        ("photo.webp", "image/webp", "WEBP"),
    ],
)
def test_image_validation_accepts_supported_formats(
    filename, content_type, image_format
):
    attachments = asyncio.run(
        validate_image_uploads(
            [_upload(filename, content_type, _image_bytes(image_format))]
        )
    )

    assert attachments[0]["content_type"] == content_type
    assert all(
        item["data_url"].startswith(f"data:{item['content_type']};base64,")
        for item in attachments
    )


def test_image_validation_accepts_multiple_files():
    attachments = asyncio.run(
        validate_image_uploads(
            [
                _upload("one.png", "image/png", _image_bytes()),
                _upload("two.webp", "image/webp", _image_bytes("WEBP")),
            ]
        )
    )

    assert [item["filename"] for item in attachments] == [
        "one.png",
        "two.webp",
    ]


@pytest.mark.parametrize(
    ("filename", "content_type", "content", "status_code"),
    [
        ("bad.svg", "image/svg+xml", b"<svg/>", 415),
        ("spoofed.png", "image/png", b"not an image", 422),
        ("mismatch.png", "image/jpeg", _image_bytes(), 415),
    ],
)
def test_image_validation_rejects_unsupported_or_malformed_files(
    filename, content_type, content, status_code
):
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            validate_image_uploads(
                [_upload(filename, content_type, content)]
            )
        )

    assert error.value.status_code == status_code


def test_image_validation_rejects_oversized_files():
    upload = _upload(
        "large.png",
        "image/png",
        b"x" * (MAX_IMAGE_SIZE_BYTES + 1),
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(validate_image_uploads([upload]))

    assert error.value.status_code == 413


def test_image_validation_rejects_more_than_three_images():
    uploads = [
        _upload("photo.png", "image/png", _image_bytes())
        for _ in range(MAX_IMAGE_COUNT + 1)
    ]
    with pytest.raises(HTTPException) as error:
        asyncio.run(validate_image_uploads(uploads))

    assert error.value.status_code == 413
