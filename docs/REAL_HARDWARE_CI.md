# Real-Hardware CI Setup

> Sprint 5 deliverable — how to register a self-hosted Snapdragon X-series
> runner so `tests/e2e/test_real_sdk_e2e.py` runs on every nightly and
> on PRs labelled `needs:real-hw`.

The mock-mode CI matrix in `.github/workflows/ci.yml` runs on
GitHub-hosted x86_64 runners and exercises ~2071 unit tests. The
real-hardware workflow in `.github/workflows/real-hw.yml` complements
it by exercising the actual `qairt-converter` / `snpe-net-run` /
`qairt-quantizer` subprocess paths on a Snapdragon X-series box.

## Hardware

Tested machines:
- Dell Latitude 7455 (Snapdragon X Elite X1E80100, Windows 11 Pro)
- Lenovo ThinkPad T14s Gen 6 (Snapdragon X Elite)
- Microsoft Surface Laptop 7

Any Snapdragon X / X Elite / X2 Elite Copilot+ PC with 16 GB+ RAM,
500 GB+ free disk, and stable network access works.

## One-time setup

### 1. Stage QAIRT archive

The runner doesn't re-download QAIRT each run (it's 1.7 GB). Place
the archive in a stable location:

```powershell
# On the runner:
$qairt = "$env:USERPROFILE\qairt\v2.46.0.260424.zip"
mkdir -Force (Split-Path $qairt) | Out-Null
# (Manually download QAIRT to that path — Qualcomm portal requires login.)
```

### 2. Register the runner

```powershell
# In the QUAD repo on github.com, go to:
#   Settings → Actions → Runners → "New self-hosted runner"
#
# Pick "Windows ARM64", then run the displayed config.cmd. Use these labels:
./config.cmd --labels "self-hosted,snapdragon-x,windows" --runasservice
```

The label tuple `[self-hosted, snapdragon-x, windows]` is what
`real-hw.yml` matches in `runs-on`.

### 3. Set the repository variable

```
Settings → Secrets and variables → Actions → Variables → New repository variable
  Name:  QAIRT_TEST_ARCHIVE
  Value: C:\Users\<runner-user>\qairt\v2.46.0.260424.zip
```

The workflow passes this to `QUAD_TEST_QAIRT_ARCHIVE`, and the e2e
test prefers that env var over auto-detecting in `~/Downloads/`.

### 4. Install Python 3.12 on the runner

```powershell
winget install Python.Python.3.12
# Verify
python --version    # Python 3.12.x
```

### 5. Test the pipeline manually before relying on CI

```powershell
git clone https://github.com/pkr465/QUAD.git
cd QUAD
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,real]"
$env:QUAD_TEST_QAIRT_ARCHIVE = "$env:USERPROFILE\qairt\v2.46.0.260424.zip"
$env:PYTHONIOENCODING = "utf-8"
pytest tests/e2e/test_real_sdk_e2e.py -v -s
```

Expected: all 7 phases pass in ~5 seconds.

## How the workflow runs

| Trigger | When | Purpose |
|---|---|---|
| `workflow_dispatch` | Manual | On-demand validation |
| `schedule` (nightly 03:00 UTC) | Nightly | Trunk health watchdog |
| PR label `needs:real-hw` | When the label is added | Gate sensitive merges |

The job runs three steps:
1. **Mock-mode unit tests** — sanity check that the venv is healthy
2. **e2e test** against the real QAIRT SDK (subprocess to `qnn-platform-validator.exe`)
3. **Real-mode `detect_hardware`** — exercises the Sprint-1 P0-1/P0-4 fix
   live; prints the parsed `DeviceProfile` JSON

If any step fails, the job fails and the maintainer team is notified.

## Cost / risk notes

- **Self-hosted runners are not isolated.** Don't allow PRs from
  forks to trigger this workflow. The `pull_request` trigger here
  uses `types: [labeled]` so an authorised maintainer must apply
  `needs:real-hw` before the runner picks up the PR.
- **Disk usage.** Each run extracts QAIRT into `./sdks/`. The
  workspace is preserved across runs, so the extraction is idempotent
  (the e2e test detects an existing install and skips re-extracting).
- **Reboots.** If the runner reboots, the GitHub Actions runner service
  starts automatically (we passed `--runasservice` above).

## Future improvements

- Record measured latency / FPS per nightly run and trend it.
- Add real `snpe-net-run` inference to the e2e test using a small
  ONNX model (post-Sprint-3 the compiler can produce a `.dlc` from
  `mobilenetv2.onnx` automatically).
- Mirror the matrix on a Snapdragon 8 Elite Android device via ADB —
  see `docs/REAL_HARDWARE.md` for the connection plumbing.
