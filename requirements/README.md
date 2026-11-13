# EIDF experiment environments

`eidf-pytorch.txt` and `eidf-bleurt.txt` record the packages installed for the
successful 13 November 2026 GPU experiment. They were copied from the saved
`uv pip freeze` outputs and checked against the scoring workers' package
metadata. The machine-specific editable-project entry is omitted; the setup
script installs the current checkout separately.

Use these files through `scripts/bootstrap_eidf.sh`, which selects Python
3.12.16 and the CUDA 12.8 PyTorch wheel index. The two environments intentionally
have different NVIDIA library versions. TensorFlow's library search path is
set only for TensorFlow processes. The driver came with the EIDF VM image
(570.124.06 on the original A100); this script does not install a driver.

These Linux GPU requirements complement the project's general `uv.lock`.
They pin installed dependency versions, but are not wheel-hash lockfiles and
do not pin Ubuntu packages. The bootstrap records `dpkg-query -W` for each
new setup. Model revisions are recorded separately in each run's `models.json`.
