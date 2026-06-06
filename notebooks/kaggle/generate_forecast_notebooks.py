#!/usr/bin/env python3
"""Generate 8 per-(session, mode) DPAD forecast notebooks for parallel Kaggle runs.

Runs inline in Kaggle's native Python 3.12 (no conda subprocess) so TF gets
direct GPU access. Kaggle GPU kernels ship TF with CUDA already configured.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

SESSIONS = ['PDI1_S2', 'PDI1_S4', 'PDI4_S2', 'PDI4_S3']
MODES    = ['z-as-behavior', 'z-as-neural']
H_VALUES = [5.0]

def slug(session, mode):
    s = session.lower().replace('_', '-')
    return f'{s}-{mode}'

def kernel_id(session, mode):
    return f'giedriusmirklys/dpad-forecast-{slug(session, mode)}'

def nb_filename(session, mode):
    return f'dpad_forecast_{session}_{mode}.ipynb'

def meta_filename(session, mode):
    return f'kernel-metadata-forecast-{session}-{mode}.json'

# ── Cell templates ────────────────────────────────────────────────────────────

def cell_md(session, mode):
    return (f'# DPAD Forecast — {session} / {mode}\n\n'
            f'H_VALUES={H_VALUES}\n\n'
            f'Runs inline in Kaggle native Python 3.12 for direct GPU access.\n'
            f'Dataset: `giedriusmirklys/dpad-splits`')

# Install only packages not already present in Kaggle's image.
# TF + CUDA are pre-installed with GPU support — don't reinstall.
CELL_INSTALL = """\
import subprocess, sys

def pip(*args):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *args])

pip('--no-deps', '--ignore-requires-python', 'DPAD==0.0.9', 'PSID==1.2.6')
pip('polars>=1.0.0', 'pyyaml', 'pyarrow', 'statsmodels')

import tensorflow as tf
print('TF version:', tf.__version__)
print('GPUs:', tf.config.list_physical_devices('GPU'))"""

CELL_SETUP = """\
import os, sys, re
from pathlib import Path

WORK = Path('/kaggle/working')

for _candidate in [
    Path('/kaggle/input/datasets/giedriusmirklys/dpad-splits'),
    Path('/kaggle/input/dpad-splits'),
]:
    if _candidate.exists():
        DATA = _candidate
        break
else:
    raise SystemExit('dpad-splits dataset not mounted')
print('DATA:', DATA)

# add project src to path
SRC = str(DATA / 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.chdir(str(WORK))

# symlink existing results + setups into writable working dir
(WORK/'results'/'dpad').mkdir(parents=True, exist_ok=True)
for sv in sorted(DATA.glob('results/dpad/dpad_*_dbs_*')):
    dst = WORK/'results'/'dpad'/sv.name
    dst.mkdir(parents=True, exist_ok=True)
    for item in sv.iterdir():
        lnk = dst/item.name
        if not lnk.exists():
            os.symlink(item, lnk)
    (dst/'forecast').mkdir(exist_ok=True)

dst_setups = WORK/'training'/'setups'/'dpad_modal'
dst_setups.mkdir(parents=True, exist_ok=True)
for y in (DATA/'training/setups/dpad_modal').glob('*.yaml'):
    lnk = dst_setups/y.name
    if not lnk.exists():
        os.symlink(y, lnk)

(WORK/'logs'/'dpad').mkdir(parents=True, exist_ok=True)
(WORK/'tmp_cfgs').mkdir(exist_ok=True)
print(f'staged {len(list((WORK/"results"/"dpad").iterdir()))} variant dirs')"""

def cell_run(session, mode):
    return f"""\
import re, tarfile, tempfile
from pathlib import Path

# patch DPADWrapper.predict to avoid m-step model rebuild between predict/forecast
from utils.frameworks import dpad as _dmod
_cache = _dmod._DPADFWK_FORECAST_CACHE
_orig_predict = _dmod.DPADWrapper.predict
import numpy as np

def _fast_predict(self, Y, Z=None):
    _prev_m = _cache.get(id(self.idSys))
    if _prev_m is not None:
        # already in multi-step mode — extract 1-step from preds[0], preds[m], preds[2m]
        bs = self.idSys.block_samples
        all_Zp, all_Yp, all_Xp = [], [], []
        for y in Y:
            n = y.shape[0]
            rem = n % bs
            yp = np.concatenate([y, np.zeros((bs-rem, y.shape[1]))]) if rem else y
            preds = self.idSys.predict(yp)
            all_Zp.append(np.asarray(preds[0])[:n])
            all_Yp.append(np.asarray(preds[_prev_m])[:n])
            all_Xp.append(np.asarray(preds[2*_prev_m])[:n])
        return all_Zp, all_Yp, all_Xp
    return _orig_predict(self, Y, Z)

_dmod.DPADWrapper.predict = _fast_predict

WORK     = Path('/kaggle/working')
SESSION  = {repr(session)}
MODE     = {repr(mode)}
H_VALUES = {repr(H_VALUES)}
YAML_DIR = WORK / 'training' / 'setups' / 'dpad_modal'
TMP      = WORK / 'tmp_cfgs'

_H_RE = re.compile(r'( {{4}}h_grid:\\n)(?:    - \\S+\\n)+')

def patch_yaml(text, hv):
    blk = '    h_grid:\\n' + ''.join(f'    - {{h}}\\n' for h in hv)
    return _H_RE.sub(blk, text)

def checkpoint(h):
    hd = f'h{{h:g}}'
    tar_path = WORK / f'forecast_{{SESSION}}_{{MODE}}_{{hd}}.tar.gz'
    written = []
    with tarfile.open(tar_path, 'w:gz') as tar:
        for rd in sorted(WORK.glob(f'results/dpad/dpad_{{MODE}}_{{SESSION}}_*_dbs_both/')):
            fd = rd / 'forecast' / hd
            if fd.exists():
                tar.add(fd, arcname=str(fd.relative_to(WORK)))
                written.append(rd.name)
    print(f'checkpoint {{tar_path.name}}: {{tar_path.stat().st_size/1e6:.1f}} MB, dirs={{written}}')

from utils.config import get_config
from utils.logger import setup_logging
from training.pipelines._base import FrameworkPipeline

yp = YAML_DIR / f'dpad_modal_{{SESSION}}_{{MODE}}.yaml'
if not yp.exists():
    raise SystemExit(f'YAML missing: {{yp}}')

errors = []
for h in H_VALUES:
    hd = f'h{{h:g}}'
    print(f'\\n=== {{SESSION}} {{MODE}} h={{h}} ===')
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', dir=TMP,
                                     prefix=f'dpad_{{SESSION}}_{{MODE}}_',
                                     delete=False) as tmp:
        tmp.write(patch_yaml(yp.read_text(), [h]))
        tmp_path = Path(tmp.name)
    try:
        config = get_config(str(tmp_path))
        log = setup_logging(
            f'dpad_{{SESSION}}_{{MODE}}_{{hd}}',
            WORK / 'logs' / 'dpad' / f'{{SESSION}}_{{MODE}}_{{hd}}.log')
        FrameworkPipeline(config, log, phases=('forecasts',)).run()
        print(f'DONE h={{h}} — checkpointing...')
        checkpoint(h)
    except Exception:
        import traceback; traceback.print_exc()
        errors.append(h)
    finally:
        tmp_path.unlink(missing_ok=True)

print(f'\\nFinished. {{len(errors)}} errors: {{errors}}')"""

def cell_verify(session, mode):
    return f"""\
from pathlib import Path
WORK = Path('/kaggle/working')
for h in {repr(H_VALUES)}:
    hd = f'h{{h:g}}'
    hits = sorted(WORK.glob(f'results/dpad/dpad_{mode}_{session}_*_dbs_both/forecast/{{hd}}/test/*.parquet'))
    print(f'{{hd}} test: {{"OK" if hits else "MISSING"}} ({{len(hits)}} parquets)')
tars = sorted(WORK.glob('forecast_*.tar.gz'))
print(f'checkpoints: {{[t.name for t in tars]}}')"""

# ── Notebook builder ──────────────────────────────────────────────────────────

def make_code_cell(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

def make_md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def make_notebook(session, mode):
    return {
        "cells": [
            make_md_cell(cell_md(session, mode)),
            make_code_cell(CELL_INSTALL),
            make_code_cell(CELL_SETUP),
            make_code_cell(cell_run(session, mode)),
            make_code_cell(cell_verify(session, mode)),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def make_metadata(session, mode):
    s = session.replace('_', ' ')
    m = mode.replace('-', ' ')
    title = f"DPAD Forecast {s} {m}"
    return {
        "id": kernel_id(session, mode),
        "title": title,
        "code_file": nb_filename(session, mode),
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": ["giedriusmirklys/dpad-splits"],
        "competition_sources": [],
        "kernel_sources": [],
    }

# ── Generate ──────────────────────────────────────────────────────────────────

for session in SESSIONS:
    for mode in MODES:
        nb_path   = HERE / nb_filename(session, mode)
        meta_path = HERE / meta_filename(session, mode)
        nb_path.write_text(json.dumps(make_notebook(session, mode), indent=1))
        meta_path.write_text(json.dumps(make_metadata(session, mode), indent=2))
        print(f'wrote {nb_path.name}')

print(f'\nDone. {len(SESSIONS) * len(MODES)} notebooks generated.')
