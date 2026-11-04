import base64
import io
import warnings
from collections.abc import Sequence

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

MAX_IMAGE_COUNT = 3
MAX_IMAGE_SIZE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_IMAGE_SIZE_BYTES = MAX_IMAGE_COUNT * MAX_IMAGE_SIZE_BYTES
MAX_IMAGE_PIXELS = 40_000_000

_SUPPORTED_IMAGES = {
    ".png": ("PNG", "image/png"),
    ".jpg": ("JPEG", "image/jpeg"),
    ".jpeg": ("JPEG", "image/jpeg"),
    ".webp": ("WEBP", "image/webp"),
}


def _validate_image_bytes(
    filename: str,
    content_type: str,
    image_bytes: bytes,
) -> dict[str, str]:
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    expected = _SUPPORTED_IMAGES.get(f".{extension}")
    if expected is None or content_type.lower() != expected[1]:
        raise HTTPException(
            status_code=415,
            detail="Use PNG, JPG, JPEG, or WEBP images.",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as image:
                image_format = image.format
                width, height = image.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise HTTPException(
                        status_code=413,
                        detail="Image dimensions exceed the supported limit.",
                    )
                image.verify()

            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
    except HTTPException:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        UnidentifiedImageError,
    ) as exc:
        raise HTTPException(
            status_code=422,
            detail="The uploaded file is not a valid, readable image.",
        ) from exc

    if image_format != expected[0]:
        raise HTTPException(
            status_code=415,
            detail="The image contents do not match the file type.",
        )

    safe_filename = filename.replace("\\", "/").rsplit("/", 1)[-1][:255]
    return {
        "filename": safe_filename or f"image.{extension}",
        "content_type": expected[1],
        "data_url": (
            f"data:{expected[1]};base64,"
            f"{base64.b64encode(image_bytes).decode('ascii')}"
        ),
    }


async def validate_image_uploads(
    uploads: Sequence[UploadFile],
) -> list[dict[str, str]]:
    if len(uploads) > MAX_IMAGE_COUNT:
        raise HTTPException(
            status_code=413,
            detail=f"Attach no more than {MAX_IMAGE_COUNT} images.",
        )

    attachments: list[dict[str, str]] = []
    total_size = 0
    for upload in uploads:
        filename = upload.filename or ""
        content_type = upload.content_type or ""
        image_bytes = await upload.read(MAX_IMAGE_SIZE_BYTES + 1)
        if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Each image must be 4 MB or smaller.",
            )
        total_size += len(image_bytes)
        if total_size > MAX_TOTAL_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="The combined image size exceeds the supported limit.",
            )

        attachments.append(
            await run_in_threadpool(
                _validate_image_bytes,
                filename,
                content_type,
                image_bytes,
            )
        )

    return attachments
