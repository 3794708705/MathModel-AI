from io import BytesIO
from pathlib import Path

from mathmodel_ai.data.profiler import DataProfiler
from mathmodel_ai.files.parsers import ParserRegistry
from mathmodel_ai.files.pipeline import FilePipeline
from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.files.validation import FileValidator


def pipeline_for(root: Path) -> FilePipeline:
    return FilePipeline(
        store=LocalFileStore(root),
        validator=FileValidator(
            max_archive_entries=100,
            max_archive_uncompressed_bytes=10 * 1024 * 1024,
            max_archive_ratio=50,
            max_image_pixels=1_000_000,
        ),
        parsers=ParserRegistry(max_pdf_pages=20, max_pdf_page_images=2),
        profiler=DataProfiler(),
        max_upload_bytes=10 * 1024 * 1024,
        max_multimodal_inline_bytes=5 * 1024 * 1024,
    )


def stream(data: bytes) -> BytesIO:
    return BytesIO(data)
