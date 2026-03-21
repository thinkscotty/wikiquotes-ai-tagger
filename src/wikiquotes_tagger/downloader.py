"""Download and decompress Wikiquote XML dumps."""

from __future__ import annotations

import bz2
from pathlib import Path

import click
import httpx

DUMP_DIRECT_URL = (
    "https://dumps.wikimedia.org/enwikiquote/latest/"
    "enwikiquote-latest-pages-articles.xml.bz2"
)

CHUNK_SIZE = 1024 * 1024  # 1 MB


def download_dump(data_dir: Path) -> Path:
    """Download the latest Wikiquote dump to data_dir. Returns path to .bz2 file.

    Skips download if file already exists and size matches the remote Content-Length.
    Uses streaming download with a click progress bar.
    """
    bz2_path = data_dir / "enwikiquote-latest-pages-articles.xml.bz2"

    # Check if we can skip the download
    if bz2_path.exists():
        # Quick HEAD request to check size
        try:
            head = httpx.head(DUMP_DIRECT_URL, follow_redirects=True, timeout=30.0)
            remote_size = int(head.headers.get("content-length", 0))
            local_size = bz2_path.stat().st_size
            if remote_size > 0 and local_size == remote_size:
                click.echo(f"Dump already downloaded: {bz2_path} ({local_size / 1e6:.0f} MB)")
                return bz2_path
        except httpx.HTTPError:
            pass  # Can't check, re-download to be safe

    click.echo(f"Downloading Wikiquote dump from {DUMP_DIRECT_URL}")

    with httpx.stream(
        "GET",
        DUMP_DIRECT_URL,
        follow_redirects=True,
        timeout=httpx.Timeout(30.0, read=None),
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(bz2_path, "wb") as f:
            with click.progressbar(length=total, label="Downloading") as bar:
                for chunk in response.iter_bytes(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    bar.update(len(chunk))

    size_mb = bz2_path.stat().st_size / 1e6
    click.echo(f"Download complete: {size_mb:.0f} MB")
    return bz2_path


def decompress_dump(bz2_path: Path) -> Path:
    """Decompress .bz2 to .xml using streaming reads.

    Skips if .xml already exists and is newer than .bz2.
    """
    xml_path = bz2_path.with_suffix(".xml")

    if xml_path.exists() and xml_path.stat().st_mtime > bz2_path.stat().st_mtime:
        click.echo(f"XML already decompressed: {xml_path}")
        return xml_path

    click.echo("Decompressing (this may take a few minutes)...")

    bz2_size = bz2_path.stat().st_size

    with open(bz2_path, "rb") as raw_f, open(xml_path, "wb") as dst:
        decompressor = bz2.BZ2Decompressor()
        with click.progressbar(length=bz2_size, label="Decompressing") as bar:
            while True:
                chunk = raw_f.read(CHUNK_SIZE)
                if not chunk:
                    break
                try:
                    decompressed = decompressor.decompress(chunk)
                    if decompressed:
                        dst.write(decompressed)
                except EOFError:
                    break
                bar.update(len(chunk))

    return xml_path


def find_dump_file(data_dir: Path) -> Path | None:
    """Find an existing .xml dump file in data_dir."""
    for p in sorted(data_dir.glob("*.xml"), key=lambda x: x.stat().st_mtime, reverse=True):
        return p
    return None
