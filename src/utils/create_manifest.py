"""
Generate a snapshot manifest JSON from DVC pointers.
"""
import argparse, base64, datetime as dt, glob, hashlib, json, os, subprocess, sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Please pip install pyyaml", file=sys.stderr); sys.exit(1)


def git_commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def iso_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_dvc_md5s(dvc_file: Path) -> str:
    """Collect md5 from *.dvc pointer file"""
    with open(dvc_file) as f:
        try:
            data = yaml.safe_load(f)
            outs = data.get("outs") or []
            if outs:
                md = outs[0].get("md5")
                return md
        except Exception as e:
            print(f"[warn] skip {p}: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dvc-file", default= "../data/cancer_data.dvc", help="Path to a file that contains .dvc pointers ")
    ap.add_argument("--out", default= "../../snapshots/manifest.json", help="Manifest JSON path to write (e.g., snapshots/latest.json)")
    ap.add_argument("--tag-prefix", default="features-", help="Prefix for the snapshot tag (e.g., features-)")
    ap.add_argument("--tag", default="", help="Explicit tag suffix (e.g., 2025-08-18). If empty, uses YYYY-MM-DD UTC.")
    args = ap.parse_args()

    dvc_dir = Path(args.dvc_file)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # collect md5s
    md5s = read_dvc_md5s(dvc_dir)

    today = args.tag or dt.datetime.utcnow().date().isoformat()
    tag = f"{args.tag_prefix}{today}"
    commit_sha = git_commit_sha()

    manifest = {
        "tag": tag,
        "commit_sha": commit_sha,
        "dvc_md5": md5s,
        "created_at": iso_now(),
    }
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out_path}:\n{json.dumps(manifest, indent=2)}")


if __name__ == "__main__":
    main()
