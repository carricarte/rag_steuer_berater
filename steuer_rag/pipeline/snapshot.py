"""Persist the Chroma vector index to an HF Dataset repo between container restarts.

Flow:
  restore_snapshot() — download + extract tarball from HF Dataset → chroma_dir
  upload_snapshot()  — compress chroma_dir → upload tarball to HF Dataset

The dataset repo is created automatically (private) on first upload.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)
_FILENAME = "chroma_snapshot.tar.gz"


def upload_snapshot(chroma_dir: Path, dataset_repo: str, token: str) -> None:
    """Compress chroma_dir and push it as a single tarball to a private HF Dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=dataset_repo, repo_type="dataset", exist_ok=True, private=True)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as fh:
        tmp = Path(fh.name)
    try:
        with tarfile.open(tmp, "w:gz") as tar:
            tar.add(chroma_dir, arcname="chroma")
        log.info("[snapshot] compressed %.1f MB → uploading to %s", tmp.stat().st_size / 1e6, dataset_repo)
        api.upload_file(
            path_or_fileobj=str(tmp),
            path_in_repo=_FILENAME,
            repo_id=dataset_repo,
            repo_type="dataset",
        )
        log.info("[snapshot] upload complete")
    finally:
        tmp.unlink(missing_ok=True)


def restore_snapshot(chroma_dir: Path, dataset_repo: str, token: str | None = None) -> bool:
    """Download and extract snapshot from HF Dataset. Returns True on success."""
    from huggingface_hub import hf_hub_download

    try:
        local_path = hf_hub_download(
            repo_id=dataset_repo,
            filename=_FILENAME,
            repo_type="dataset",
            token=token,
        )
    except Exception as exc:
        log.info("[snapshot] snapshot not available in %s: %s", dataset_repo, exc)
        return False

    try:
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)
        chroma_dir.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(local_path, "r:gz") as tar:
            _safe_extractall(tar, chroma_dir.parent)

        extracted = chroma_dir.parent / "chroma"
        if extracted.exists() and extracted != chroma_dir:
            extracted.rename(chroma_dir)

        log.info("[snapshot] restored to %s", chroma_dir)
        return True
    except Exception as exc:
        log.warning("[snapshot] extraction failed: %s", exc)
        shutil.rmtree(chroma_dir, ignore_errors=True)
        return False


def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    dest_str = str(dest.resolve())
    for member in tar.getmembers():
        if not str((dest / member.name).resolve()).startswith(dest_str):
            raise ValueError(f"Unsafe tar path rejected: {member.name}")
    tar.extractall(dest)
