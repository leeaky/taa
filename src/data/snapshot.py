# =============================================================================
# DATA SNAPSHOT
# Creates a committed snapshot of live fetched data for use on Streamlit Cloud
# where live FRED/Yahoo fetching is not available on a schedule.
#
# Usage (run locally after runner.py has populated the cache):
#   python src/data/snapshot.py
#
# This copies the current data cache into src/data/snapshot/ which is
# committed to git. The cloud dashboard reads from this snapshot instead
# of attempting live fetches.
#
# Refresh the snapshot whenever you want to update the cloud showcase
# with more recent data:
#   1. python src/runner.py           (fetch latest data)
#   2. python src/data/snapshot.py   (commit snapshot)
#   3. git add src/data/snapshot/
#   4. git commit -m "Update data snapshot YYYY-MM-DD"
#   5. git push
# =============================================================================

import sys
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime

SRC = Path(__file__).parent.parent
sys.path.insert(0, str(SRC))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

SNAPSHOT_DIR = SRC / "data" / "snapshot"


def create_snapshot(force: bool = False):
    """
    Copy the current data cache into data/snapshot/.
    Raises if cache doesn't exist unless force=True.
    """
    import config
    cache_dir = config.DATA_DIR

    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Cache not found at {cache_dir}. "
            f"Run 'python src/runner.py' first to populate the cache."
        )

    # Check cache has content
    parquet_files = list(cache_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"Cache at {cache_dir} is empty. "
            f"Run 'python src/runner.py' first."
        )

    # Check cache age
    meta_path = cache_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        # Get most recent fetched_at
        fetched_ats = [v["fetched_at"] for v in meta.values() if "fetched_at" in v]
        if fetched_ats:
            latest = max(fetched_ats)
            age_hours = (datetime.utcnow() - datetime.fromisoformat(latest)).total_seconds() / 3600
            if age_hours > 48 and not force:
                logger.warning(
                    f"Cache is {age_hours:.0f}h old. "
                    f"Consider running 'python src/runner.py' first to refresh. "
                    f"Use --force to snapshot anyway."
                )

    # Create snapshot directory
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy all parquet files and meta
    copied = []
    for f in cache_dir.iterdir():
        if f.suffix in (".parquet", ".json"):
            dest = SNAPSHOT_DIR / f.name
            shutil.copy2(f, dest)
            copied.append(f.name)
            logger.info(f"Copied: {f.name}")

    # Write snapshot metadata
    snapshot_meta = {
        "created_at": datetime.utcnow().isoformat(),
        "source":     "live_fetch",
        "files":      copied,
        "note":       "Committed data snapshot for Streamlit Cloud showcase. "
                      "Run src/data/snapshot.py to refresh.",
    }
    with open(SNAPSHOT_DIR / "snapshot_meta.json", "w") as f:
        json.dump(snapshot_meta, f, indent=2)

    logger.info(f"Snapshot created: {len(copied)} files in {SNAPSHOT_DIR}")
    logger.info(f"Snapshot date: {snapshot_meta['created_at'][:10]}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  git add src/data/snapshot/")
    logger.info(f"  git commit -m 'Update data snapshot {datetime.utcnow().strftime('%Y-%m-%d')}'")
    logger.info("  git push")

    return SNAPSHOT_DIR


def load_snapshot() -> dict | None:
    """
    Load data from the committed snapshot.
    Returns None if snapshot doesn't exist.
    Used by the fetcher as a fallback on Streamlit Cloud.
    """
    if not SNAPSHOT_DIR.exists():
        return None

    meta_path = SNAPSHOT_DIR / "meta.json"
    if not meta_path.exists():
        return None

    try:
        import pandas as pd

        with open(meta_path) as f:
            meta = json.load(f)

        results = {}
        for sid, info in meta.items():
            safe = sid.replace("^", "_")
            pq = SNAPSHOT_DIR / f"{safe}.parquet"
            if pq.exists():
                s = pd.read_parquet(pq).squeeze()
                results[sid] = {
                    "data":       s,
                    "source":     info["source"] + "_SNAPSHOT",
                    "fetched_at": datetime.fromisoformat(info["fetched_at"]),
                    "frequency":  info["frequency"],
                }

        snap_meta_path = SNAPSHOT_DIR / "snapshot_meta.json"
        if snap_meta_path.exists():
            with open(snap_meta_path) as f:
                snap_meta = json.load(f)
            logger.info(
                f"Loaded snapshot from {snap_meta.get('created_at', 'unknown')[:10]}. "
                f"{len(results)} series."
            )

        return results if results else None

    except Exception as e:
        logger.warning(f"Could not load snapshot: {e}")
        return None


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    try:
        create_snapshot(force=force)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
