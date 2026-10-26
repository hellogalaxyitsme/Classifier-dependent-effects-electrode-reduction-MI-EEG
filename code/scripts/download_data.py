"""Download raw recordings of the confirmatory datasets (no decoding is run).

usage: python download_data.py <DatasetClassName>
Writes progress to stdout; already present files are skipped by MOABB.
"""
import sys
import time
import traceback

import moabb
import moabb.datasets as md

name = sys.argv[1]
ds = getattr(md, name)()
print(f"{time.strftime('%F %T')} {name}: moabb {moabb.__version__}, {len(ds.subject_list)} subjects", flush=True)
failed = []
for s in ds.subject_list:
    t0 = time.time()
    for attempt in range(3):
        try:
            ds.data_path(s)
            print(f"{time.strftime('%F %T')} {name} subject {s} ok ({time.time() - t0:.0f}s)", flush=True)
            break
        except Exception as exc:
            print(f"{time.strftime('%F %T')} {name} subject {s} attempt {attempt} failed: {exc!r}"[:400], flush=True)
            traceback.print_exc(limit=2)
            time.sleep(30)
    else:
        failed.append(s)
print(f"{time.strftime('%F %T')} {name} DONE; failed subjects: {failed}", flush=True)
