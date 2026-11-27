"""Direct HTTP/HTTPS file download with strict SSRF protection on every redirect hop."""
import ipaddress
from pathlib import Path
import re
import socket
from typing import Optional, Callable, Tuple
from urllib.parse import urlparse, unquote, urljoin

import httpx

from .base import SourceAdapter
from ..schemas import SourceInfo

ALLOWED_MEDIA_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mp3", ".wav", ".flac", ".m4a"}
MAX_REDIRECTS = 5


class SSRFError(Exception):
    pass


def validate_public_ip(host: str) -> str:
    """Resolves host and verifies that ALL resolved IPs are public, non-reserved addresses."""
    if not host:
        raise SSRFError("URL has no hostname.")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise SSRFError(f"Could not resolve host '{host}': {e}")

    resolved_ips = []
    for _, _, _, _, sockaddr in infos:
        ip_str = sockaddr[0]
        ip = ipaddress.ip_address(ip_str)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise SSRFError(f"Access to private or local network IP '{ip_str}' is forbidden.")
        resolved_ips.append(ip_str)

    if not resolved_ips:
        raise SSRFError("No valid public IP found.")
    return resolved_ips[0]


def _safe_request(client: httpx.Client, method: str, url: str, headers: dict, stream: bool = False) -> Tuple[httpx.Response, str]:
    """Executes HTTP request following redirects manually while re-validating every hop."""
    cur_url = url
    for _ in range(MAX_REDIRECTS + 1):
        parsed = urlparse(cur_url)
        if parsed.scheme not in ("http", "https"):
            raise SSRFError(f"Unsupported URL scheme '{parsed.scheme}'.")
        if parsed.username or parsed.password:
            raise SSRFError("Credentials in URLs are not permitted.")
        validate_public_ip(parsed.hostname or "")

        req = client.build_request(method, cur_url, headers=headers)
        resp = client.send(req, stream=stream)
        if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("location"):
            nxt = urljoin(cur_url, resp.headers["location"])
            resp.close()
            cur_url = nxt
            continue
        return resp, cur_url
    raise SSRFError("Too many redirects.")


class DirectUrlAdapter(SourceAdapter):
    def can_handle(self, url_or_path: str) -> bool:
        if not url_or_path.startswith(("http://", "https://")):
            return False
        path = unquote(urlparse(url_or_path).path)
        ext = Path(path).suffix.lower()
        return ext in ALLOWED_MEDIA_EXTS or bool(re.search(r"\.(mp4|webm|mov|mkv|mp3|wav|m4a)(\?|$)", url_or_path, re.I))

    def probe(self, url_or_path: str) -> SourceInfo:
        try:
            with httpx.Client(timeout=15.0, follow_redirects=False) as client:
                resp, final_url = _safe_request(
                    client, "GET", url_or_path,
                    {"Range": "bytes=0-1024", "User-Agent": "MarhabaClipMind/3.0"},
                    stream=True,
                )
                try:
                    if resp.status_code >= 400:
                        return SourceInfo(
                            url_or_path=url_or_path,
                            source_type="direct_url",
                            platform="direct",
                            is_supported=False,
                            error_message=f"HTTP server returned status {resp.status_code}",
                        )
                    ctype = resp.headers.get("content-type", "").lower()
                    if ctype.startswith("text/html"):
                        return SourceInfo(
                            url_or_path=url_or_path,
                            source_type="direct_url",
                            platform="direct",
                            is_supported=False,
                            error_message="Link points to an HTML web page rather than a media file.",
                        )
                    name = Path(unquote(urlparse(final_url).path)).name or "downloaded_video.mp4"
                    return SourceInfo(
                        url_or_path=url_or_path,
                        source_type="direct_url",
                        platform="direct",
                        is_supported=True,
                        title=Path(name).stem,
                    )
                finally:
                    resp.close()
        except SSRFError as e:
            return SourceInfo(
                url_or_path=url_or_path,
                source_type="direct_url",
                platform="direct",
                is_supported=False,
                error_message=str(e),
            )
        except Exception as e:
            return SourceInfo(
                url_or_path=url_or_path,
                source_type="direct_url",
                platform="direct",
                is_supported=False,
                error_message=f"Connection failed: {e}",
            )

    def fetch_media(self, url_or_path: str, dest_dir: Path, audio_only: bool = False,
                    progress_cb: Optional[Callable[[float, str], None]] = None) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = Path(unquote(urlparse(url_or_path).path)).name or "video.mp4"
        dest_path = dest_dir / name

        if progress_cb:
            progress_cb(0.1, f"Downloading {name}...")

        with httpx.Client(timeout=60.0, follow_redirects=False) as client:
            resp, _ = _safe_request(client, "GET", url_or_path, {"User-Agent": "MarhabaClipMind/3.0"}, stream=True)
            try:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length") or 0)
                downloaded = 0
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=128 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0 and progress_cb:
                            pct = min(0.95, downloaded / total)
                            progress_cb(pct, f"Downloaded {downloaded // (1024*1024)} MB / {total // (1024*1024)} MB")
            finally:
                resp.close()

        if progress_cb:
            progress_cb(1.0, f"Saved {dest_path.name}")
        return dest_path


ALLOWED_EXTENSIONS_SET = ALLOWED_MEDIA_EXTS
