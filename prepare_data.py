"""Fetch the original AMdataset images from the pinned authors' release."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1
from http.client import HTTPException
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


REVISION = "52fe75dec70c1e8046493b4e3256f8f752a4d39d"
REPO = "mahsan2/AMdataset"
CLASSES = ("badcylinder", "goodcylinder")


def fetch(url):
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": "AMdataset-reproduction"})
            with urlopen(request, timeout=90) as response:
                return response.read()
        except (OSError, HTTPException):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def blob_hash(data):
    return sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def download(entry, root):
    relative = Path(entry["path"])
    target = root / relative
    if target.is_file() and blob_hash(target.read_bytes()) == entry["sha"]:
        return
    url = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{quote(relative.as_posix())}"
    data = fetch(url)
    if blob_hash(data) != entry["sha"]:
        raise ValueError(f"Git blob checksum mismatch: {relative}")
    pending = target.with_suffix(".part")
    pending.write_bytes(data)
    pending.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("AMdataset"))
    args = parser.parse_args()
    tree_url = f"https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1"
    tree = json.loads(fetch(tree_url))
    if tree["truncated"] or tree["sha"] != REVISION:
        raise ValueError("GitHub returned an incomplete or unexpected tree.")
    entries = []
    for category in CLASSES:
        directory = Path("dataset") / category
        (args.root / directory).mkdir(parents=True, exist_ok=True)
        matches = [e for e in tree["tree"] if e["type"] == "blob"
                   and Path(e["path"]).parent == directory
                   and Path(e["path"]).suffix.lower() == ".jpg"]
        if not matches:
            raise ValueError(f"No images listed for {category}")
        print(f"{category}: {len(matches)} images", flush=True)
        entries.extend(matches)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for index, _ in enumerate(pool.map(lambda e: download(e, args.root), entries), 1):
            if index % 25 == 0 or index == len(entries):
                print(f"Verified {index}/{len(entries)}", flush=True)


if __name__ == "__main__":
    main()
