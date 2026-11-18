#!/usr/bin/env python3
"""Check reference links in every Markdown file of this repo.

Rules (each violation is printed and makes the exit status 1):
  arXiv   - must be an abs link with a version: https://arxiv.org/abs/2104.09864v5
          - the version must exist, and the link text must equal the paper title from the arXiv API
            (case and whitespace ignored), so a wrong ID cannot hide behind a plausible title
  GitHub  - a blob/tree link must be pinned: a 40-hex commit or a tag that looks like a version (v1.2.3, 1.2)
            never a branch name such as main/master
          - the pinned URL must answer HTTP 200
Run with --offline to apply only the format rules (no network).

Usage: scripts/check_links.py [--offline] [paths...]   (default: every *.md under the repo root)
"""
from __future__ import annotations

import datetime
import email.utils
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
ARXIV_ANY = re.compile(r"https?://(?:www\.)?arxiv\.org/(abs|pdf|html)/([0-9]{4}\.[0-9]{4,5})(v[0-9]+)?(?:\.pdf)?/?$")
GH_PIN = re.compile(r"https://github\.com/[^/]+/[^/]+/(blob|tree)/([^/]+)(/.*)?$")
GH_ANY = re.compile(r"https?://(?:www\.)?github\.com/", re.IGNORECASE)
SHA = re.compile(r"^[0-9a-f]{40}$")
TAG = re.compile(r"^v?[0-9]+(\.[0-9]+)*([.-]?[A-Za-z0-9]+)*$")
ATOM = "{http://www.w3.org/2005/Atom}"
UA = {"User-Agent": "llm-papers-link-check/1.0"}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def retry_after(value: str | None) -> float:
    """Seconds asked for by a Retry-After header, either delta-seconds or an HTTP date; 0 if absent or unreadable."""
    if not value:
        return 0.0
    if value.strip().isdigit():
        return float(value.strip())
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return 0.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.timezone.utc)
    return max(0.0, (when - datetime.datetime.now(datetime.timezone.utc)).total_seconds())


def arxiv_title(aid: str, version: str) -> str | None:
    """Title of that exact version, or None if the version does not exist."""
    url = f"https://export.arxiv.org/api/query?id_list={aid}{version}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                root = ET.fromstring(r.read())
            break
        except urllib.error.HTTPError as e:
            # The API rate-limits with 429 or 503; wait and retry instead of failing the whole run.
            # Any other status, or the fifth refusal, is raised and fails the check.
            if e.code not in (429, 503) or attempt == 4:
                raise
            time.sleep(retry_after(e.headers.get("Retry-After")) or 15 * 2**attempt)
        except (TimeoutError, urllib.error.URLError):
            if attempt == 4:
                raise
            time.sleep(15 * 2**attempt)
    for entry in root.findall(f"{ATOM}entry"):
        eid = entry.findtext(f"{ATOM}id") or ""
        if eid.rstrip("/").endswith(f"{aid}{version}"):
            return entry.findtext(f"{ATOM}title")
    return None


def http_ok(url: str) -> bool:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except urllib.error.HTTPError:
        return False


def check_file(path: Path, offline: bool, errors: list[str]) -> int:
    n = 0
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for text, url in LINK.findall(line):
            where = f"{path.relative_to(REPO) if path.is_relative_to(REPO) else path}:{lineno}"
            m = ARXIV_ANY.match(url)
            if "arxiv.org" in url.lower():
                n += 1
                if not m or not url.startswith("https://arxiv.org/abs/") or not m.group(3):
                    errors.append(f"{where}: arXiv link must be https://arxiv.org/abs/<id>v<N>: {url}")
                    continue
                if offline:
                    continue
                aid, ver = m.group(2), m.group(3)
                title = arxiv_title(aid, ver)
                time.sleep(3)  # arXiv API asks for at most one request every 3 seconds
                if title is None:
                    errors.append(f"{where}: arXiv {aid}{ver} does not exist")
                elif norm(title) != norm(text):
                    errors.append(f"{where}: title mismatch for {aid}{ver}\n    link text: {text}\n    arXiv:     {norm(title)}")
            elif GH_ANY.match(url):
                if not url.startswith("https://github.com/"):
                    n += 1
                    errors.append(f"{where}: GitHub link must start with https://github.com/: {url}")
                    continue
                g = GH_PIN.match(url)
                if not g:
                    continue  # repo/org/issue/PR links are fine unpinned
                n += 1
                ref = g.group(2)
                if not (SHA.match(ref) or (TAG.match(ref) and any(c.isdigit() for c in ref))):
                    errors.append(f"{where}: GitHub link not pinned to a commit or version tag ({ref}): {url}")
                    continue
                if not offline and not http_ok(url.split("#")[0]):
                    errors.append(f"{where}: GitHub permalink does not answer 200: {url}")
    return n


def main(argv: list[str]) -> int:
    offline = "--offline" in argv
    targets = [Path(a).resolve() for a in argv if a != "--offline"]
    files = targets or sorted(p for p in REPO.rglob("*.md") if ".venv" not in p.parts)
    errors: list[str] = []
    checked = sum(check_file(f, offline, errors) for f in files)
    for e in errors:
        print("FAIL", e)
    print(f"{len(files)} files, {checked} pinned links checked ({'offline' if offline else 'online'}), {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
