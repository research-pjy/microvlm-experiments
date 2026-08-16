# microvlm (NanoVLM experiments)

Iteration-0 research codebase for reproducing and extending
[NanoVLMs](https://arxiv.org/abs/2502.07838) (Agarwalla et al., 2025).
Implementation lives in the installable `microvlm` package under `src/`.
Notebooks only call `microvlm.experiments.*` and plot results.

## Environment

A conda environment named **`microvlm` already exists**. Do not create a new
one. On Ubuntu:

```bash
conda activate microvlm
make setup    # fails loudly if a different (or no) conda env is active
```

`make setup` runs `pip install -e ".[dev]"` inside that env.
`environment.yml` documents dependencies for reproducibility; it is not the
install path.

## External COCO fixture (required before real tests)

This repo does **not** bundle COCO and does **not** sample it.
Populate `tests/fixtures/` *externally* with the standalone script
`sample_coco_fixture.py` (that script is **not** part of this repository).
Drop the result here before running integration tests:

```
tests/fixtures/
    images/          # e.g. 000000000139.jpg
    captions.json    # dict keyed by COCO image_id string
```

See `tests/fixtures/README.md` for the fixed schema. If the fixture is missing,
integration tests skip with a pointer to that README instead of failing.

## Ollama (local Ubuntu and DGX)

Both machines run Ollama. Model **names are never hardcoded** in backend
classes; they come from YAML:

- teachers: `configs/teacher/local_ollama.yaml`
- judges: `configs/evaluation/judge_local_ollama.yaml`

Local smoke (small models), DGX (larger models) use the same code:

```bash
# Ubuntu (example)
ollama serve
ollama pull llama3.2:3b

# DGX: pull a larger tag, then set `model:` in the YAML above.
```

Default host is `http://localhost:11434` (overridable in the same YAML).

## Smoke test

```bash
conda activate microvlm
make test          # unit tests; integration skips without fixture/Ollama
make smoke         # end-to-end if fixture + Ollama are available
```

Open `notebooks/00_environment_smoke_test.ipynb` and run all cells: it asserts
the `microvlm` conda env, builds a mini model with encoder arch_a, runs one
forward pass, and prints the parameter count (on the order of Table 1's 5M).

Full-scale training is **not** run on this Ubuntu machine. Use
`training=local_smoke` locally (CPU, ≤5 images) and `training=dgx_full` on the
DGX. A CPU run with more than 5 samples raises `RuntimeError` on purpose.

## Visual encoder: why arch_a and arch_b both exist

Section 2.2.1 of the paper describes splitting the image into 16×16 patches and
then applying Conv2D per patch. Figure 4 instead shows two Conv2D layers on the
full image producing feature maps, and Figure 5 looks like a standard ViT patch
embedding. Those artifacts cannot all be implemented as one network. This repo
keeps **two swappable encoders** and does not invent a third:

- `arch_a` (`configs/model/encoder/arch_a_patch_then_conv.yaml`): patch, then
  conv — literal reading of Section 2.2.1. In the research notes this is
  “Architecture B”.
- `arch_b` (`configs/model/encoder/arch_b_conv_then_patch.yaml`): conv on the
  whole image, then tokens — Figure 4 plus strided-conv patch embedding. In the
  research notes this is “Architecture A”.

Select them with Hydra (`encoder=arch_a_patch_then_conv` or
`encoder=arch_b_conv_then_patch`). Experiment 07 trains both on the 5-image
fixture and compares loss curves.
