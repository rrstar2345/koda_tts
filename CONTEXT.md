# Kan-TTS Rust Port — Context & Task List

## Source
- Repo: https://github.com/rrstar2345/py_kan_tts_lite.git (Python, PyTorch VITS/MMS-TTS inference for Kannada)
- Entry point: `example.py` -> `VitsModel.from_pretrained` + `VitsTokenizer.from_pretrained`, runs `model(**inputs).waveform`, writes `output.wav`.
- Model files used: `models/config.json`, `models/vocab.json`, `models/tokenizer_config.json`, `models/model.safetensors` (assumed always present — no `pytorch_model.bin` fallback needed).
- `models/config.json` key values (this checkpoint): `num_speakers=1`, `speaker_embedding_size=0`, `use_stochastic_duration_prediction=true`, `vocab_size=75`, Kannada char-level tokenizer, `phonemize=false`, `is_uroman=false`.

## Design decisions
- Crate: `candle-core` + `candle-nn` (native Rust tensor/NN lib, safetensors support, no libtorch dependency) — chosen over `tch` to avoid requiring a libtorch install.
- Single binary crate `kan_tts_rs` producing a CLI (text -> `output.wav`), plus a lib target so pieces are reusable.
- `model.rs` (1375 lines, largest file) split into multiple modules under `src/model/`:
  - `src/model/mod.rs` — `VitsModel` (top-level), `VitsModelOutput`
  - `src/model/attention.rs` — `VitsAttention` (relative-position multi-head attention)
  - `src/model/wavenet.rs` — `VitsWaveNet`, `fused_add_tanh_sigmoid_multiply`
  - `src/model/flows.rs` — `VitsResidualCouplingLayer`, `VitsResidualCouplingBlock`, `VitsConvFlow`, `VitsElementwiseAffine`, `VitsDilatedDepthSeparableConv`, spline functions (`unconstrained_rational_quadratic_spline`, `rational_quadratic_spline`)
  - `src/model/duration_predictor.rs` — `VitsStochasticDurationPredictor` (only variant needed; see removals)
  - `src/model/text_encoder.rs` — `VitsTextEncoder`, `VitsEncoder`, `VitsEncoderLayer`, `VitsFeedForward`
  - `src/model/hifigan.rs` — `VitsHifiGan`, `HifiGanResidualBlock`
- `config.rs` — `VitsConfig::from_pretrained` (reads config.json)
- `tokenizer.rs` — `VitsTokenizer` (char-level encode, blank-interspersion; matches Kannada checkpoint: normalize on, phonemize off, is_uroman off)
- `main.rs` — CLI: load config/tokenizer/model, run inference, write WAV (no external WAV crate — write RIFF header manually, mirrors Python's `wave` module usage)

## Removed vs. Python (dead code for this checkpoint, or training-only)
- `pytorch_model.bin` fallback in weight loading — removed per instructions; `model.safetensors` assumed always present.
- `VitsDurationPredictor` (non-stochastic variant) — checkpoint always has `use_stochastic_duration_prediction=true`; only `VitsStochasticDurationPredictor` is ported. Config flag kept as a validation assertion (error if false) rather than a runtime branch, since we only support this checkpoint shape.
- `VitsPosteriorEncoder` — Python comment states "used only for training"; never called in `forward`. Not ported (no inference use), so its weights are simply not loaded (safe: `from_pretrained`-equivalent loader ignores unknown/unused safetensors keys).
- Multi-speaker branches (`embed_speaker`, all `speaker_embedding_size != 0` conditionals in WaveNet/HifiGan/DurationPredictor/DilatedDepthSeparableConv) — checkpoint has `num_speakers=1`, `speaker_embedding_size=0`. Ported as `Option<Tensor>` global-conditioning params that are always `None` for this checkpoint; branch code kept minimal (no dead `nn.Embedding` speaker table).
- `phonemize=True` / `is_uroman=True` code paths in tokenizer — checkpoint config has both `false`. Ported as a startup validation error if config ever sets them true (rather than fully implementing phonemizer/uroman), matching Python's own "not bundled" behavior.
- `output_attentions` / `output_hidden_states` / `attentions` plumbing — inference-only CLI never requests these; dropped to simplify signatures. Only `last_hidden_state`/final outputs are threaded through.
- `GradientCheckpointingLayer` (empty base class), `ModelOutput`/dataclass generic base, `torch_compilable_check` (debug assertion helper) — Rust uses plain structs + `Result`/`assert!`, no need for the Python OOP scaffolding.
- `remove_weight_norm` / `apply_weight_norm` (training/export utilities, never called in inference path) — not ported.
- `create_bidirectional_mask` — VITS text encoder always runs full non-causal attention over a single non-padded sequence in this CLI (batch size 1, no padding); implemented as a simpler additive mask builder, same math, less generality.

## Task List
- [x] Clone and inspect Python source
- [x] Identify dead/unused code to drop
- [x] Set up Rust crate skeleton (Cargo.toml, module layout)
- [x] Port `config.rs` (VitsConfig + from_pretrained)
- [x] Port `tokenizer.rs`
- [x] Port `model/wavenet.rs`
- [x] Port `model/flows.rs` (coupling layers, conv flow, spline math, dilated depthwise conv)
- [x] Port `model/duration_predictor.rs` (stochastic variant only)
- [x] Port `model/attention.rs`
- [x] Port `model/text_encoder.rs` (encoder layer, feed-forward, encoder, text encoder)
- [x] Port `model/hifigan.rs`
- [x] Port `model/mod.rs` (VitsModel top-level forward)
- [x] Port `main.rs` (CLI wiring + WAV writer)
- [x] Write Cargo.toml with candle deps
- [x] User runs `cargo check` / `cargo build` and reports back any compile errors for fixes

## Notes for follow-up
- Not run `cargo check`/`cargo build` per instructions — user will run manually.
- If candle's weight-norm-parametrized Conv1d weights in safetensors are stored as `weight_g`/`weight_v` (PyTorch `weight_norm` parametrization naming), the loader recomputes `weight = weight_g * weight_v / ||weight_v||` — implemented in `model/wavenet.rs` and `model/hifigan.rs` helper `load_weight_norm_conv1d`. Verify actual safetensors key names once `cargo check`/inspection of the file is possible; may need adjusting key suffix (`.weight_g` vs `.parametrizations.weight.original0`, depending on which PyTorch weight_norm API produced the checkpoint).