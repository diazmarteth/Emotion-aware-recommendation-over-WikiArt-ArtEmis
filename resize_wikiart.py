"""Stream WikiArt images out of the kagglehub archive, downscaled, aspect preserved.

Short side -> --short-side, no cropping: stores strictly more than CLIP consumes,
so crop strategy (center / squash / pad) stays a choice at inference time.
"""
import io, os, sys, shutil, zipfile, argparse
from pathlib import Path
from multiprocessing import Pool
from PIL import Image

DEFAULT_ARCHIVE = Path.home() / ".cache/kagglehub/datasets/steubk/wikiart/1.archive"
EXTS = (".jpg", ".jpeg", ".png")
_zf = None


def _init(archive):
    global _zf
    _zf = zipfile.ZipFile(archive)


def _one(job):
    name, out_dir, short_side, quality = job
    dst = Path(out_dir) / name
    try:
        im = Image.open(io.BytesIO(_zf.read(name)))
        im.draft("RGB", None)                     # fast DCT-scaled JPEG decode
        im = im.convert("RGB")
        w, h = im.size
        s = short_side / min(w, h)
        if s < 1.0:                               # downscale only, never upscale
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        im.save(tmp, "JPEG", quality=quality, optimize=True)
        os.replace(tmp, dst)                      # atomic: no truncated survivors
        return (True, name, None)
    except Exception as e:
        return (False, name, repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    ap.add_argument("--out", default="wikiart_256")
    ap.add_argument("--short-side", type=int, default=256)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--min-free-gb", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="stop after N (smoke test)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(a.archive) as zf:
        names = [n for n in zf.namelist()
                 if n.lower().endswith(EXTS) and not n.startswith("__MACOSX")]
        for meta in ("classes.csv", "wclasses.csv"):
            try:
                (out / meta).write_bytes(zf.read(meta))
            except KeyError:
                pass

    todo = [n for n in names if not (out / n).exists()]      # resumable
    already = len(names) - len(todo)
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(names)} images in archive | {already} already done | "
          f"{len(todo)} to do | {a.workers} workers", flush=True)
    if not todo:
        print("nothing to do"); return

    jobs = ((n, str(out), a.short_side, a.quality) for n in todo)
    ok = fail = 0
    errlog = out / "_failed.log"
    with errlog.open("a") as elog, Pool(a.workers, _init, (a.archive,)) as pool:
        for i, (good, name, err) in enumerate(pool.imap_unordered(_one, jobs, chunksize=32), 1):
            if good:
                ok += 1
            else:
                fail += 1
                elog.write(f"{name}\t{err}\n"); elog.flush()
            if i % 1000 == 0:
                free_gb = shutil.disk_usage(out).free / 2**30
                print(f"{i}/{len(todo)}  ok={ok} fail={fail}  free={free_gb:.1f}GB", flush=True)
                if free_gb < a.min_free_gb:
                    print(f"ABORT: free space {free_gb:.1f}GB < {a.min_free_gb}GB. "
                          f"Re-run to resume.", flush=True)
                    pool.terminate()
                    break

    print(f"DONE ok={ok} fail={fail}" + (f" (see {errlog})" if fail else ""), flush=True)


if __name__ == "__main__":
    main()
