"""
GitHub repo cloner — clones a repo and enumerates CUDA files for batch translation.

Used by the /translate-repo endpoint to handle multi-file CUDA codebases.
"""
from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from loguru import logger


def parse_github_url(url: str) -> tuple[str, str]:
    """Parse a GitHub URL or 'owner/repo' shorthand.

    Returns (owner, repo) tuple.
    Raises ValueError if URL is invalid.
    """
    pattern = r"github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/|$)"
    match = re.search(pattern, url)
    if match:
        return match.groups()

    if "/" in url and not url.startswith("http"):
        parts = url.split("/")
        if len(parts) == 2:
            return parts[0], parts[1]

    raise ValueError(f"Invalid GitHub URL: {url}")


async def clone_repo(
    owner: str,
    repo: str,
    target_dir: Optional[Path] = None,
    depth: int = 1,
) -> Path:
    """Clone a GitHub repo (shallow clone)."""
    if target_dir is None:
        target_dir = Path(tempfile.mkdtemp(prefix="crossfire_repo_"))
    else:
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    clone_url = f"https://github.com/{owner}/{repo}.git"
    clone_path = target_dir / repo

    logger.info(f"Cloning {owner}/{repo} to {clone_path}")

    cmd = ["git", "clone", "--depth", str(depth), clone_url, str(clone_path)]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

    if proc.returncode != 0:
        error_msg = stderr.decode(errors="replace")
        raise RuntimeError(f"Git clone failed: {error_msg}")

    return clone_path


def find_cuda_files(
    repo_path: Path,
    pattern: str = "**/*.cu",
    exclude_dirs: Optional[set] = None,
) -> List[Path]:
    """Find all CUDA files in a repo."""
    if exclude_dirs is None:
        exclude_dirs = {".git", "node_modules", "build", "dist", ".cache"}

    repo_path = Path(repo_path)
    files = []

    for f in repo_path.glob(pattern):
        if any(part in exclude_dirs for part in f.parts):
            continue
        if f.stat().st_size > 100_000:
            continue
        files.append(f)

    return sorted(files)


def cleanup_repo(repo_path: Path) -> None:
    """Remove a cloned repo directory."""
    shutil.rmtree(repo_path, ignore_errors=True)


async def fetch_repo_contents(
    repo_url: str,
    file_pattern: str = "**/*.cu",
) -> dict:
    """Clone repo, find CUDA files, return metadata + sources."""
    owner, repo = parse_github_url(repo_url)
    clone_path = await clone_repo(owner, repo)
    cuda_files = find_cuda_files(clone_path, file_pattern)

    files_data = []
    for f in cuda_files:
        try:
            source = f.read_text(errors="replace")
            files_data.append({
                "path": str(f.relative_to(clone_path)),
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "source": source,
            })
        except Exception as e:
            logger.warning(f"Failed to read {f}: {e}")

    return {
        "owner": owner,
        "repo": repo,
        "clone_path": str(clone_path),
        "file_count": len(files_data),
        "files": files_data,
    }
