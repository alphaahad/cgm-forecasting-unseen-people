"""Download the three CGM datasets used in this project.

CGMacros (PhysioNet, CC BY-NC-SA 4.0): the zip and the data dictionaries, checked against PhysioNet's SHA256SUMS, then unzipped.

BUT CGM (PhysioNet, CC BY 4.0): everything EXCEPT the raw ECG/ACC .h5 files and CPET raw data.
Shanghai T1DM/T2DM (figshare collection 6310860, CC BY 4.0): all files in the collection.

Files already present with the right size/checksum are skipped, so it is safe to re-run.
"""
import hashlib
import json
import os
import re
import sys
import urllib.request

DATA = os.environ.get("MBZUAI_DATA", r"D:\MBZUAI_data")
UA = {"User-Agent": "mbzuai-cgm-research-download/1.0"}


def get(url, binary=False):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        b = r.read()
    return b if binary else b.decode("utf-8", "ignore")


def download(url, dest, expect_size=None):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and (expect_size is None or os.path.getsize(dest) == expect_size):
        return "skip"
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    return "ok"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def but_cgm():
    base = "https://physionet.org"
    root = f"{base}/files/but-cgm/1.0.0"
    out = os.path.join(DATA, "but_cgm")
    page = get(f"{base}/content/but-cgm/1.0.0/")
    subjects = sorted(set(re.findall(r'data-dfp-dir="(\d{6})"', page)))
    print("BUT CGM subjects:", len(subjects), flush=True)
    files = re.findall(r'href="(/files/but-cgm/1\.0\.0/[^"/]+)\?download"', page)
    for s in subjects:
        sp = get(f"{base}/content/but-cgm/1.0.0/{s}/")
        files += re.findall(rf'href="(/files/but-cgm/1\.0\.0/{s}/[^"]+)\?download"', sp)
    keep = [f for f in sorted(set(files)) if not re.search(r"_(ECG|ACC)\.h5$|_CPET_data\.csv$", f)]
    print(f"BUT CGM files to fetch: {len(keep)} of {len(set(files))}", flush=True)
    from concurrent.futures import ThreadPoolExecutor

    def one(f):
        rel = f[len("/files/but-cgm/1.0.0/"):]
        for attempt in range(3):
            try:
                return rel, download(f"{base}{f}?download", os.path.join(out, *rel.split("/")))
            except Exception as e:  # retry transient network errors
                err = str(e)
        return rel, f"FAILED {err}"

    with ThreadPoolExecutor(max_workers=5) as ex:  # modest parallelism, polite to the server
        for i, (rel, st) in enumerate(ex.map(one, keep), 1):
            if i % 50 == 0 or i == len(keep) or st.startswith("FAILED"):
                print(f"  {i}/{len(keep)} {rel} [{st}]", flush=True)
    # verify against PhysioNet checksums
    sums = {}
    for line in get(f"{root}/SHA256SUMS.txt").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            sums[parts[1].strip()] = parts[0]
    bad = missing = checked = 0
    for f in keep:
        rel = f[len("/files/but-cgm/1.0.0/"):]
        if rel not in sums:
            missing += 1
            continue
        checked += 1
        if sha256(os.path.join(out, *rel.split("/"))) != sums[rel]:
            bad += 1
            print("  CHECKSUM MISMATCH:", rel, flush=True)
    print(f"BUT CGM verify: checked={checked} bad={bad} not_in_sums={missing}", flush=True)


def shanghai():
    out = os.path.join(DATA, "shanghai")
    arts = json.loads(get("https://api.figshare.com/v2/collections/6310860/articles?page_size=100"))
    print("Shanghai articles:", [(a["id"], a["title"]) for a in arts], flush=True)
    for a in arts:
        info = json.loads(get(f"https://api.figshare.com/v2/articles/{a['id']}"))
        print(" license:", info.get("license", {}).get("name"), "|", info["title"], flush=True)
        for fl in info["files"]:
            dest = os.path.join(out, str(a["id"]), fl["name"])
            st = download(fl["download_url"], dest, fl.get("size"))
            ok = md5(dest) == fl.get("computed_md5")
            print(f"  {fl['name']} size={fl.get('size')} [{st}] md5_ok={ok}", flush=True)
            if fl["name"].endswith(".zip") and not os.path.isdir(os.path.join(out, "extracted")):
                import zipfile
                with zipfile.ZipFile(dest) as z:
                    z.extractall(os.path.join(out, "extracted"))
                print("  unzipped to", os.path.join(out, "extracted"), flush=True)


def cgmacros():
    import zipfile
    base = "https://physionet.org"
    root = f"{base}/files/cgmacros/1.0.0"
    out = os.path.join(DATA, "cgmacros")
    page = get(f"{base}/content/cgmacros/1.0.0/")
    files = sorted(set(re.findall(r'href="(/files/cgmacros/1\.0\.0/[^"/]+)\?download"', page)))
    print("CGMacros files:", [f.split("/")[-1] for f in files], flush=True)
    for f in files:
        name = f.split("/")[-1]
        print(f"  {name} [{download(f'{base}{f}?download', os.path.join(out, name))}]", flush=True)
    sums = {}
    for line in get(f"{root}/SHA256SUMS.txt").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            sums[parts[1].strip()] = parts[0]
    for name, digest in sums.items():
        path = os.path.join(out, name)
        if name == "LICENSE.txt":
            continue                      # PhysioNet serves an empty LICENSE.txt; the licence is CC BY-NC-SA 4.0 (see the dataset page)
        status = "OK" if os.path.exists(path) and sha256(path) == digest else "MISSING OR MISMATCH"
        print(f"  checksum {name}: {status}", flush=True)
    target = os.path.join(out, "extracted")
    if not os.path.isdir(os.path.join(target, "CGMacros")):
        with zipfile.ZipFile(os.path.join(out, "CGMacros_dateshifted365.zip")) as z:
            z.extractall(target)
        print("  unzipped to", target, flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["cgmacros", "shanghai", "but"]
    if "cgmacros" in which:
        cgmacros()
    if "shanghai" in which:
        shanghai()
    if "but" in which:
        but_cgm()
