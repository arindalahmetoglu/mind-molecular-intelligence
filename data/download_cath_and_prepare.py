#!/usr/bin/env python3
import argparse
import contextlib
import ftplib
import hashlib
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Optional, List

import requests
import re
import gzip
import shutil


LATEST_DIR = "https://download.cathdb.info/cath/releases/latest-release"
FTP_HOST = "orengoftp.biochem.ucl.ac.uk"
FTP_BASE = "/cath/releases/latest-release"


def md5sum(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_http(url: str, out_path: Path, verify_ssl: bool = True, max_retries: int = 8, backoff_seconds: int = 5) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    attempt = 0
    while True:
        attempt += 1
        try:
            resume_from = tmp.stat().st_size if tmp.exists() else 0
        except Exception:
            resume_from = 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from > 0 else {}
        with requests.get(url, stream=True, timeout=120, verify=verify_ssl, headers=headers) as r:
            # Accept 206 (partial) when resuming; 200 when starting fresh
            if r.status_code not in (200, 206):
                r.raise_for_status()
            mode = "ab" if resume_from > 0 and r.status_code == 206 else "wb"
            with open(tmp, mode) as f:
                for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
        # If we got here without exception, move into place
        os.replace(tmp, out_path)
        return
        
        # Retry on network errors
        # (Note: Flow never reaches here due to return above unless exceptions)
        # This is kept for clarity
        
        if attempt >= max_retries:
            raise
        time.sleep(backoff_seconds)


def download_ftp(host: str, remote_path: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    dir_name, file_name = os.path.split(remote_path)
    with ftplib.FTP(host, timeout=120) as ftp:
        ftp.set_pasv(True)
        ftp.login()
        if dir_name:
            ftp.cwd(dir_name)
        with open(tmp, "wb") as f:
            ftp.retrbinary(f"RETR {file_name}", f.write)
    os.replace(tmp, out_path)


def ftp_listdir(host: str, remote_dir: str) -> list[str]:
    with ftplib.FTP(host, timeout=60) as ftp:
        ftp.login()
        ftp.cwd(remote_dir)
        entries: list[str] = []
        ftp.retrlines("NLST", entries.append)
        return entries


def http_dir_candidates(base_url: str, subset: str) -> List[str]:
    """Parse an HTTP directory listing and return candidate filenames for the subset."""
    url = f"{base_url}/non-redundant-data-sets/"
    # Prefer plain HTTP
    url = url.replace("https://", "http://")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    text = r.text
    pattern = re.compile(rf"cath-dataset-nonredundant-{subset}-[^\s\"']+?\.pdb\.tgz")
    candidates = sorted(set(pattern.findall(text)))
    return candidates


def safe_extract(tar_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            prefix = os.path.commonprefix([abs_directory, abs_target])
            return prefix == abs_directory

        for member in tar.getmembers():
            member_path = os.path.join(dest_dir, member.name)
            if not is_within_directory(dest_dir, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
        tar.extractall(dest_dir)


def main():
    parser = argparse.ArgumentParser(description="Download CATH non-redundant PDB set and prepare for training")
    parser.add_argument("--subset", choices=["S20", "S40"], default="S40", help="CATH non-redundant subset")
    parser.add_argument("--version_tag", default=None, help="CATH version tag used in filenames (auto if omitted)")
    parser.add_argument("--out-root", default="/home/arin/mind_esa/data/cath/latest", help="Where to store downloaded/extracted data")
    parser.add_argument("--link-raw", default="/home/arin/mind_esa/data/pdb_dataset/raw", help="Where to symlink .pdb files for the loader")
    parser.add_argument("--no_extract", action="store_true", help="Skip extraction")
    parser.add_argument("--no_verify", action="store_true", help="Skip md5 verification")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Determine filename. If version_tag not provided, discover via HTTP HEAD/FTP listing
    if args.version_tag:
        fname = f"cath-dataset-nonredundant-{args.subset}-{args.version_tag}.pdb.tgz"
    else:
        # Try simple HTTP probing across known tags and base URLs
        probe_tags = ["v4_4_0", "v4_3_0", "v4_2_0"]
        base_candidates = [
            "http://download.cathdb.info/cath/releases/latest-release",
            "http://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release",
            "http://download.cathdb.info/cath/releases/all-releases/v4_4_0",
            "http://orengoftp.biochem.ucl.ac.uk/cath/releases/all-releases/v4_4_0",
            "http://download.cathdb.info/cath/releases/all-releases/v4_3_0",
            "http://orengoftp.biochem.ucl.ac.uk/cath/releases/all-releases/v4_3_0",
        ]
        found: Optional[str] = None
        for base in base_candidates:
            for tag in probe_tags:
                test_name = f"cath-dataset-nonredundant-{args.subset}-{tag}.pdb.tgz"
                # latest-release uses non-redundant-data-sets; all-releases paths include tag dir
                if base.endswith("latest-release"):
                    test_url = f"{base}/non-redundant-data-sets/{test_name}"
                else:
                    test_url = f"{base}/non-redundant-data-sets/{test_name}"
                try:
                    r = requests.head(test_url, timeout=20, allow_redirects=True)
                    if r.status_code == 200:
                        found = (test_name, base)
                        print(f"Found tarball via HTTP HEAD: {test_name} at {base}")
                        raise StopIteration
                except StopIteration:
                    break
                except Exception:
                    pass
            if found:
                break
        if not found:
            # Last resort: FTP listing
            try:
                nr_dir = f"{FTP_BASE}/non-redundant-data-sets"
                entries = ftp_listdir(FTP_HOST, nr_dir)
                candidates = [e for e in entries if e.startswith(f"cath-dataset-nonredundant-{args.subset}-") and e.endswith(".pdb.tgz")]
                candidates.sort()
                if candidates:
                    found = (candidates[-1], LATEST_DIR)
                    print(f"Found tarball via FTP listing: {found[0]}")
            except Exception:
                pass
        if not found:
            print("Could not determine CATH tarball name. Specify --version_tag explicitly (e.g., v4_3_0).")
            sys.exit(1)
        fname, discovered_base = found

    md5_fname = fname + ".md5"
    # If we discovered a different base, prefer it
    base_for_http = discovered_base if 'discovered_base' in locals() else LATEST_DIR
    http_url = f"{base_for_http}/non-redundant-data-sets/{fname}"
    http_md5_url = f"{base_for_http}/non-redundant-data-sets/{md5_fname}"
    ftp_path = f"{FTP_BASE}/non-redundant-data-sets/{fname}"
    ftp_md5_path = f"{FTP_BASE}/non-redundant-data-sets/{md5_fname}"

    tar_path = out_root / fname
    md5_path = out_root / md5_fname

    # Download tarball
    if not tar_path.exists() or tar_path.stat().st_size == 0:
        # Prefer plain HTTP to avoid TLS hostname issues
        http_url_plain = http_url.replace("https://", "http://")
        try:
            print(f"Downloading {fname} via HTTP...")
            download_http(http_url_plain, tar_path, verify_ssl=False)
        except Exception as e_http:
            print(f"HTTP failed: {e_http}. Trying HTTPS relaxed...")
            try:
                download_http(http_url, tar_path, verify_ssl=False)
            except Exception as e_relaxed:
                print(f"HTTPS relaxed failed: {e_relaxed}. Falling back to FTP...")
                download_ftp(FTP_HOST, ftp_path, tar_path)
    else:
        print(f"Found existing tar: {tar_path}")

    # Download md5 (best-effort). Some mirrors do not provide .md5 for all-releases.
    md5_available = False
    if not md5_path.exists() or md5_path.stat().st_size == 0:
        print(f"Downloading MD5 {md5_fname} (best-effort)...")
        try:
            download_http(http_md5_url.replace("https://", "http://"), md5_path, verify_ssl=False)
            md5_available = True
        except Exception:
            try:
                download_http(http_md5_url, md5_path, verify_ssl=False)
                md5_available = True
            except Exception as e_md5:
                print(f"Warning: MD5 fetch failed ({e_md5}); proceeding without remote checksum")
    else:
        print(f"Found existing md5: {md5_path}")
        md5_available = True

    # Verify md5
    if not args.no_verify and md5_available:
        print("Verifying MD5 checksum...")
        ref = None
        try:
            with open(md5_path, "r") as f:
                line = f.read().strip()
                ref = line.split()[0]
        except Exception:
            pass
        if ref:
            calc = md5sum(tar_path)
            if calc.lower() != ref.lower():
                print(f"MD5 mismatch: expected {ref}, got {calc}")
                sys.exit(2)
            print("MD5 OK")
        else:
            print("Warning: could not parse MD5 file; skipping strict verification")
    elif not md5_available:
        print("Warning: MD5 not available from server; skipped verification")

    # Extract
    if not args.no_extract:
        extract_dir = out_root / args.subset
        if not extract_dir.exists() or not any(extract_dir.glob("**/*.pdb")):
            print(f"Extracting to {extract_dir}...")
            safe_extract(tar_path, extract_dir)
        else:
            print(f"Found existing extraction: {extract_dir}")

        # Symlink (or decompress then link) into raw folder for the loader
        raw_dir = Path(args.link_raw)
        raw_dir.mkdir(parents=True, exist_ok=True)
        count_linked = 0
        count_decompressed = 0

        def safe_unlink(p: Path) -> None:
            try:
                if p.exists() or p.is_symlink():
                    p.unlink()
            except Exception:
                pass

        # 1) Direct .pdb
        for src in extract_dir.rglob("*.pdb"):
            target = raw_dir / src.name
            safe_unlink(target)
            try:
                os.symlink(src, target)
                count_linked += 1
            except FileExistsError:
                pass

        # 2) .pdb.gz -> decompress to .pdb
        for src in extract_dir.rglob("*.pdb.gz"):
            dest = raw_dir / src.name.replace(".pdb.gz", ".pdb")
            if not dest.exists():
                try:
                    with gzip.open(src, 'rb') as fin, open(dest, 'wb') as fout:
                        shutil.copyfileobj(fin, fout)
                    count_decompressed += 1
                except Exception:
                    # Skip on failure
                    pass

        # 3) .ent or .ent.gz (rename to .pdb)
        for src in extract_dir.rglob("*.ent"):
            dest = raw_dir / (src.stem + ".pdb")
            if not dest.exists():
                try:
                    os.symlink(src, dest)
                    count_linked += 1
                except FileExistsError:
                    pass
        for src in extract_dir.rglob("*.ent.gz"):
            dest = raw_dir / (src.stem.replace('.ent', '') + ".pdb")
            if not dest.exists():
                try:
                    with gzip.open(src, 'rb') as fin, open(dest, 'wb') as fout:
                        shutil.copyfileobj(fin, fout)
                    count_decompressed += 1
                except Exception:
                    pass

        # 4) .cif or .cif.gz (optional; some loaders can read mmCIF)
        for src in extract_dir.rglob("*.cif"):
            target = raw_dir / src.name
            safe_unlink(target)
            try:
                os.symlink(src, target)
                count_linked += 1
            except FileExistsError:
                pass
        for src in extract_dir.rglob("*.cif.gz"):
            dest = raw_dir / src.name.replace(".cif.gz", ".cif")
            if not dest.exists():
                try:
                    with gzip.open(src, 'rb') as fin, open(dest, 'wb') as fout:
                        shutil.copyfileobj(fin, fout)
                    count_decompressed += 1
                except Exception:
                    pass

        # 5) CATH dompdb files without extensions -> link as .pdb
        dompdb_dir = extract_dir / "dompdb"
        if dompdb_dir.exists():
            for src in dompdb_dir.iterdir():
                if src.is_file():
                    name = src.name
                    if "." not in name:  # no extension
                        dest = raw_dir / f"{name}.pdb"
                        safe_unlink(dest)
                        try:
                            os.symlink(src, dest)
                            count_linked += 1
                        except FileExistsError:
                            pass

        print(f"Linked {count_linked} files; decompressed {count_decompressed} files into {raw_dir}")

    print("Done.")


if __name__ == "__main__":
    main()


