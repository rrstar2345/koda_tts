//! Port of `src/vits/config.py`.
//!
//! Only the fields actually read by the model/tokenizer code are kept.
//! Serde defaults mirror the Python dataclass defaults; `from_pretrained`
//! loads `config.json` the same way the Python `VitsConfig.from_pretrained`
//! does (unknown keys in the JSON are ignored, missing keys fall back to
//! defaults).

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct VitsConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub window_size: usize,
    pub use_bias: bool,
    pub ffn_dim: usize,
    pub ffn_kernel_size: usize,
    pub flow_size: usize,
    pub spectrogram_bins: usize,
    pub hidden_act: String,
    pub hidden_dropout: f64,
    pub attention_dropout: f64,
    pub activation_dropout: f64,
    pub layer_norm_eps: f64,
    /// This port only supports the stochastic duration predictor path
    /// (see CONTEXT.md). Validated in `VitsModel::new`.
    pub use_stochastic_duration_prediction: bool,
    pub num_speakers: usize,
    pub speaker_embedding_size: usize,
    pub upsample_initial_channel: usize,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub leaky_relu_slope: f64,
    pub depth_separable_channels: usize,
    pub depth_separable_num_layers: usize,
    pub duration_predictor_flow_bins: usize,
    pub duration_predictor_tail_bound: f64,
    pub duration_predictor_kernel_size: usize,
    pub duration_predictor_dropout: f64,
    pub duration_predictor_num_flows: usize,
    pub duration_predictor_filter_channels: usize,
    pub prior_encoder_num_flows: usize,
    pub prior_encoder_num_wavenet_layers: usize,
    pub wavenet_kernel_size: usize,
    pub wavenet_dilation_rate: usize,
    pub wavenet_dropout: f64,
    pub speaking_rate: f64,
    pub noise_scale: f64,
    pub noise_scale_duration: f64,
    pub sampling_rate: u32,
    pub pad_token_id: Option<usize>,
}

impl Default for VitsConfig {
    fn default() -> Self {
        Self {
            vocab_size: 38,
            hidden_size: 192,
            num_hidden_layers: 6,
            num_attention_heads: 2,
            window_size: 4,
            use_bias: true,
            ffn_dim: 768,
            ffn_kernel_size: 3,
            flow_size: 192,
            spectrogram_bins: 513,
            hidden_act: "relu".to_string(),
            hidden_dropout: 0.1,
            attention_dropout: 0.1,
            activation_dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_stochastic_duration_prediction: true,
            num_speakers: 1,
            speaker_embedding_size: 0,
            upsample_initial_channel: 512,
            upsample_rates: vec![8, 8, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            leaky_relu_slope: 0.1,
            depth_separable_channels: 2,
            depth_separable_num_layers: 3,
            duration_predictor_flow_bins: 10,
            duration_predictor_tail_bound: 5.0,
            duration_predictor_kernel_size: 3,
            duration_predictor_dropout: 0.5,
            duration_predictor_num_flows: 4,
            duration_predictor_filter_channels: 256,
            prior_encoder_num_flows: 4,
            prior_encoder_num_wavenet_layers: 4,
            wavenet_kernel_size: 5,
            wavenet_dilation_rate: 1,
            wavenet_dropout: 0.0,
            speaking_rate: 1.0,
            noise_scale: 0.667,
            noise_scale_duration: 0.8,
            sampling_rate: 16_000,
            pad_token_id: None,
        }
    }
}

impl VitsConfig {
    /// Loads `<model_dir>/config.json`, matching the fields understood by
    /// this port. Extra keys present in the JSON (e.g. `architectures`,
    /// `model_type`, `transformers_version`) are ignored by serde's
    /// `Deserialize` derive by default (unknown fields -> error unless we
    /// allow them), so we deserialize into a permissive intermediate map
    /// first and only keep recognized keys.
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let config_path = model_dir.as_ref().join("config.json");
        let text = std::fs::read_to_string(&config_path)
            .with_context(|| format!("reading {}", config_path.display()))?;

        // Deserialize permissively: unknown fields in config.json must not
        // cause a hard failure (mirrors Python's `valid_fields` filtering).
        let raw: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", config_path.display()))?;

        let mut cfg = VitsConfig::default();
        merge_field(&raw, "vocab_size", &mut cfg.vocab_size);
        merge_field(&raw, "hidden_size", &mut cfg.hidden_size);
        merge_field(&raw, "num_hidden_layers", &mut cfg.num_hidden_layers);
        merge_field(&raw, "num_attention_heads", &mut cfg.num_attention_heads);
        merge_field(&raw, "window_size", &mut cfg.window_size);
        merge_field(&raw, "use_bias", &mut cfg.use_bias);
        merge_field(&raw, "ffn_dim", &mut cfg.ffn_dim);
        merge_field(&raw, "ffn_kernel_size", &mut cfg.ffn_kernel_size);
        merge_field(&raw, "flow_size", &mut cfg.flow_size);
        merge_field(&raw, "spectrogram_bins", &mut cfg.spectrogram_bins);
        merge_field(&raw, "hidden_act", &mut cfg.hidden_act);
        merge_field(&raw, "hidden_dropout", &mut cfg.hidden_dropout);
        merge_field(&raw, "attention_dropout", &mut cfg.attention_dropout);
        merge_field(&raw, "activation_dropout", &mut cfg.activation_dropout);
        merge_field(&raw, "layer_norm_eps", &mut cfg.layer_norm_eps);
        merge_field(
            &raw,
            "use_stochastic_duration_prediction",
            &mut cfg.use_stochastic_duration_prediction,
        );
        merge_field(&raw, "num_speakers", &mut cfg.num_speakers);
        merge_field(&raw, "speaker_embedding_size", &mut cfg.speaker_embedding_size);
        merge_field(&raw, "upsample_initial_channel", &mut cfg.upsample_initial_channel);
        merge_field(&raw, "upsample_rates", &mut cfg.upsample_rates);
        merge_field(&raw, "upsample_kernel_sizes", &mut cfg.upsample_kernel_sizes);
        merge_field(&raw, "resblock_kernel_sizes", &mut cfg.resblock_kernel_sizes);
        merge_field(&raw, "resblock_dilation_sizes", &mut cfg.resblock_dilation_sizes);
        merge_field(&raw, "leaky_relu_slope", &mut cfg.leaky_relu_slope);
        merge_field(&raw, "depth_separable_channels", &mut cfg.depth_separable_channels);
        merge_field(&raw, "depth_separable_num_layers", &mut cfg.depth_separable_num_layers);
        merge_field(&raw, "duration_predictor_flow_bins", &mut cfg.duration_predictor_flow_bins);
        merge_field(&raw, "duration_predictor_tail_bound", &mut cfg.duration_predictor_tail_bound);
        merge_field(&raw, "duration_predictor_kernel_size", &mut cfg.duration_predictor_kernel_size);
        merge_field(&raw, "duration_predictor_dropout", &mut cfg.duration_predictor_dropout);
        merge_field(&raw, "duration_predictor_num_flows", &mut cfg.duration_predictor_num_flows);
        merge_field(
            &raw,
            "duration_predictor_filter_channels",
            &mut cfg.duration_predictor_filter_channels,
        );
        merge_field(&raw, "prior_encoder_num_flows", &mut cfg.prior_encoder_num_flows);
        merge_field(
            &raw,
            "prior_encoder_num_wavenet_layers",
            &mut cfg.prior_encoder_num_wavenet_layers,
        );
        merge_field(&raw, "wavenet_kernel_size", &mut cfg.wavenet_kernel_size);
        merge_field(&raw, "wavenet_dilation_rate", &mut cfg.wavenet_dilation_rate);
        merge_field(&raw, "wavenet_dropout", &mut cfg.wavenet_dropout);
        merge_field(&raw, "speaking_rate", &mut cfg.speaking_rate);
        merge_field(&raw, "noise_scale", &mut cfg.noise_scale);
        merge_field(&raw, "noise_scale_duration", &mut cfg.noise_scale_duration);
        merge_field(&raw, "sampling_rate", &mut cfg.sampling_rate);
        merge_field(&raw, "pad_token_id", &mut cfg.pad_token_id);

        Ok(cfg)
    }
}

/// Copies `raw[key]` into `*field` if present and of the right shape;
/// otherwise leaves the default untouched (matches the Python behavior of
/// only consuming recognized, present keys).
fn merge_field<T: serde::de::DeserializeOwned>(raw: &serde_json::Value, key: &str, field: &mut T) {
    if let Some(value) = raw.get(key) {
        if let Ok(parsed) = serde_json::from_value::<T>(value.clone()) {
            *field = parsed;
        }
    }
}