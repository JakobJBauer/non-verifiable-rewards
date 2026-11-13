# Running the benchmark on the EIDF development VM

The full experiment uses 2,000 pairs per dataset, Qwen3-4B in bfloat16,
RoBERTa-large BERTScore, and BLEURT-20. It performs inference only. The local
128-pair pilot remains at Git tag `cpu-pilot-128-pairs`.

The 13 November 2026 run completed all 56,000 candidate scores without errors
on one NVIDIA A100-SXM4-40GB. Scoring and initial reporting took 33 minutes
46 seconds, excluding setup and downloads. The local mirror is
`outputs/eidf_qwen3_4b/`; start with
[the interpretation](../outputs/eidf_qwen3_4b/analysis.md) or
[the HTML report](../outputs/eidf_qwen3_4b/report.html).

## Storage layout

| Location on `eidf-dev` | Contents | Survives VM deletion? |
| --- | --- | --- |
| `/root/non-verifiable-rewards` | Active source checkout | No |
| `/root/nvr-venv` | Python 3.12, PyTorch CUDA 12.8, Transformers, BERTScore, analysis | No |
| `/root/nvr-bleurt-venv` | Python 3.12, TensorFlow CUDA libraries, official BLEURT | No |
| `/root/.cache/non-verifiable-rewards` | Active model weights and tokenizers | No |
| `/workspace/writeable/non-verifiable-rewards/outputs/qwen3_4b_20261113` | Inputs, score journals, metadata, reports, figures | **Yes** |
| `/workspace/writeable/non-verifiable-rewards/setup` | Setup logs, dependency versions, source bundle, launch scripts, tests | **Yes** |
| `/workspace/writeable/non-verifiable-rewards/model_cache` | Backup of downloaded model assets | **Yes** |

The active environments and weights use the VM's local disk for speed. Scores
are written directly to persistent storage, one complete pair per flushed
journal record. The cache backup is a separate resumable `rsync` operation.
The VM may be deleted after 90 minutes of GPU inactivity; no artificial GPU
activity is used to keep it alive.

## Environment and execution

[`scripts/bootstrap_eidf.sh`](../scripts/bootstrap_eidf.sh) is the complete
setup entry point. It consolidates the successful commands from the original
run, including the TensorFlow library-path fix. It installs and checks:

- Ubuntu packages: Git, curl, CA certificates, unzip, build tools, tmux, rsync.
- uv 0.13.8 and Python 3.12.16.
- `/root/nvr-venv`: PyTorch 2.9.1+cu128, Transformers 5.22.2,
  BERTScore 0.3.13, datasets, analysis/plotting packages, and pytest.
- `/root/nvr-bleurt-venv`: TensorFlow 2.21.0, its NVIDIA CUDA libraries,
  official BLEURT at the recorded Git commit, analysis packages, and pytest.
- GPU matrix multiplication in both frameworks and the test suite in both
  environments.
- Frozen-data preparation, model downloads, exact neural input preparation,
  and a persistent model-cache backup.

All installed Python dependency versions are in
[`requirements/eidf-pytorch.txt`](../requirements/eidf-pytorch.txt) and
[`requirements/eidf-bleurt.txt`](../requirements/eidf-bleurt.txt), checked
against the successful workers' metadata. The original VM supplied the NVIDIA
driver (570.124.06); no driver or system CUDA toolkit installation was needed.
PyTorch's wheel uses CUDA 12.8. TensorFlow and its NVIDIA packages are kept in
their own environment.

TensorFlow needs its pip-installed NVIDIA library directories on
`LD_LIBRARY_PATH` on this VM image. `scripts/run_eidf.sh` sets these only for
the BLEURT worker; it does not expose TensorFlow's CUDA libraries to PyTorch.

From this local repository, copy the current code, including the setup scripts
and dependency files that may not yet be pushed to GitHub:

```bash
ssh -x -o BatchMode=yes eidf-dev 'mkdir -p /root/non-verifiable-rewards'
rsync -az --exclude='.venv' --exclude='artifacts' --exclude='outputs' \
  --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.ruff_cache' \
  -e 'ssh -x -o BatchMode=yes' ./ eidf-dev:/root/non-verifiable-rewards/
```

This includes Git history and current uncommitted source files, while keeping
local environments and downloaded assets off the transfer. It does not delete
existing remote files. Alternatively, clone the repository on the VM after
the updated scripts and dependency files have been pushed.

Then run setup (as root, as provided by this SSH target):

```bash
ssh -t -x eidf-dev 'cd /root/non-verifiable-rewards && bash scripts/bootstrap_eidf.sh'
```

The setup may download about 13 GB of model assets plus Python/CUDA packages.
It reuses existing Python 3.12.16 environments and cached assets. Each setup
attempt writes a new timestamped directory under
`/workspace/writeable/non-verifiable-rewards/setup/`, containing its log,
hardware and package records, tests, Git bundle, and a source archive including
uncommitted files. It leaves the earlier experiment's setup records intact.
Existing frozen inputs and model revisions are reused; missing assets are
downloaded. Setup prepares the experiment but does not score it.

Launch scoring in tmux so it survives SSH disconnection:

```bash
ssh -x eidf-dev "tmux new-session -d -s nvr-benchmark 'cd /root/non-verifiable-rewards && bash scripts/run_eidf.sh'"
ssh -t -x eidf-dev 'tmux attach -t nvr-benchmark'
```

Detach with `Ctrl-b`, then `d`. Use an existing `nvr-benchmark` session if one
is already running. Setup and the launcher default to
`configs/eidf_qwen3_4b.toml`; set `NVR_CONFIG` for both commands to select another
configuration. For a new experiment, copy the config and choose a new
persistent `run_dir`. The original directory is resumable and skips scores
already completed; it does not rerun all scoring merely because it is launched
again.

The launcher prepares data, resolves immutable model revisions, preprocesses
texts, runs lexical scoring alongside Qwen, then runs BERTScore and BLEURT on
the GPU sequentially. It creates the report after all workers finish. A failed
backend is logged while other backends are still attempted. Missing or failed
scores are never fabricated; coverage and failures must be inspected.

For the initial run, the prepared 2,000-pair samples were transferred from the
local machine's cached raw datasets. Their source and content checksums are in
`data_manifest.json`; the preparation stage verified those hashes on the VM.
All neural preprocessing caps are disabled, and the Qwen context limit is
32,768 tokens. Encoder-native truncation remains explicit and reported.

To reproduce those exact sampled inputs on a machine without the persisted
run, copy the saved inputs and pinned model manifest **before running setup**:

```bash
ssh -x eidf-dev 'mkdir -p /workspace/writeable/non-verifiable-rewards/outputs/qwen3_4b_20261113'
rsync -az -e 'ssh -x -o BatchMode=yes' \
  outputs/eidf_qwen3_4b/data \
  outputs/eidf_qwen3_4b/data_manifest.json \
  outputs/eidf_qwen3_4b/models.json \
  eidf-dev:/workspace/writeable/non-verifiable-rewards/outputs/qwen3_4b_20261113/
```

The original setup freeze files and installation logs are also available
locally in `outputs/eidf_setup/`. The initial `bootstrap.log` precedes the
TensorFlow library-path correction; use the current script for installation.

Monitor progress with the persistent `logs/orchestrator.log`, individual
`logs/*-console.log`, and `logs/gpu-telemetry.csv`. Raw journals are under
`scores/<backend>/`. `completed_backend_failures.txt` records the number of
failed backends; zero indicates all backends completed without reported pair
failures. Verify actual coverage as well as that completion marker. The current
launcher closes its scoring log before generating the final artifact manifest.
Stop any external telemetry writer before the final report, and do not redirect
report output into the run directory: writing a log after it is hashed would
invalidate its checksum. Regenerate the report after adding supplementary files.

## Recovering after the VM is deleted

1. Recreate the source checkout from the persistent source bundle or GitHub.
   Use the exact scoring source used for the run; source archives are saved
   inside the run directory. Apply the saved `setup/source_notes/` overlay to
   recover the EIDF configuration, launch scripts, and supplementary analysis
   that were added after the original Git commit. New setup attempts instead
   save a complete `source-working-tree.tar.gz` alongside their Git bundle;
   overlay that archive after recreating the checkout.
2. Recreate the two environments in `/root` using the saved setup scripts and
   dependency manifests. Preserve the recorded model and scoring settings.
3. Restore the backed-up model cache to `/root/.cache/non-verifiable-rewards`,
   or use the download stage to retrieve the recorded model revisions.
4. Use the original configuration and persistent run directory. The workers
   resume complete journals, validate run identity, and repair an incomplete
   final line. A changed scoring implementation requires a new run directory.

The current bootstrap automatically restores `model_cache/` to local storage
when it exists. The equivalent manual commands are:

```bash
mkdir -p /root/.cache/non-verifiable-rewards
rsync -a /workspace/writeable/non-verifiable-rewards/model_cache/ \
  /root/.cache/non-verifiable-rewards/
```

Reports can be rebuilt from completed scores without loading GPU models:

```bash
/root/nvr-venv/bin/python -m non_verifiable_rewards.experiments report \
  --run-dir /workspace/writeable/non-verifiable-rewards/outputs/qwen3_4b_20261113
```

When copying results to another machine, preserve the original `config.json`.
Pass the new local directory using `report --run-dir` rather than rewriting
the recorded configuration.

## Supplementary analysis

After the main report has generated the analysis tables, run:

```bash
/root/nvr-venv/bin/python -m non_verifiable_rewards.experiments.supplementary \
  --run-dir /workspace/writeable/non-verifiable-rewards/outputs/qwen3_4b_20261113 \
  --pilot-dir outputs/local_cpu_int8
```

This exports paired comparisons between every method, equal-prompt weighting
sensitivity, native encoder-truncation groups, threshold-induced ties, and
comparison with the pilot on the same 128 pairs per dataset. It also checks
that full-text ROUGE-L/length scores agree with the pilot and that monotone
VR-CLI thresholds never reverse a raw likelihood ranking. Bootstrap intervals
are exploratory and do not adjust for multiple comparisons. Pilot-versus-GPU
encoder differences combine a checkpoint change and a text-policy change.

The pilot files must be present for this optional analysis; copy
`outputs/local_cpu_int8/` from the local repository if they are absent on a
fresh VM. Regenerate the report afterwards so `artifacts.json` covers the new
tables, then copy the results back locally:

```bash
rsync -az --partial -e 'ssh -x -o BatchMode=yes' \
  eidf-dev:/workspace/writeable/non-verifiable-rewards/outputs/qwen3_4b_20261113/ \
  outputs/eidf_qwen3_4b/
rsync -az --partial -e 'ssh -x -o BatchMode=yes' \
  eidf-dev:/workspace/writeable/non-verifiable-rewards/setup/ outputs/eidf_setup/
```
