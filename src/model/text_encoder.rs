//! Port of `VitsFeedForward`, `VitsEncoderLayer`, `VitsEncoder`, and
//! `VitsTextEncoder` from `model.py`.
//!
//! `output_attentions`/`output_hidden_states`/`return_dict` plumbing and
//! `GradientCheckpointingLayer`/layerdrop are dropped (see CONTEXT.md):
//! this port always runs full-precision inference over one non-padded
//! sequence and returns only the final tensors needed downstream.

use super::attention::VitsAttention;
use crate::config::VitsConfig;
use candle_core::{Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Embedding, LayerNorm, Module, VarBuilder};

// ---------------------------------------------------------------------
// Feed-forward (position-wise conv FFN)
// ---------------------------------------------------------------------

pub struct VitsFeedForward {
    conv_1: Conv1d,
    conv_2: Conv1d,
    pad_left: usize,
    pad_right: usize,
    hidden_act: String,
}

impl VitsFeedForward {
    pub fn new(config: &VitsConfig, vb: VarBuilder) -> Result<Self> {
        // Python's nn.Conv1d has no built-in padding here (padding applied
        // manually via nn.functional.pad before each conv), so kernel convs
        // themselves use padding=0.
        let conv_1 = candle_nn::conv1d(
            config.hidden_size,
            config.ffn_dim,
            config.ffn_kernel_size,
            Conv1dConfig::default(),
            vb.pp("conv_1"),
        )?;
        let conv_2 = candle_nn::conv1d(
            config.ffn_dim,
            config.hidden_size,
            config.ffn_kernel_size,
            Conv1dConfig::default(),
            vb.pp("conv_2"),
        )?;

        let (pad_left, pad_right) = if config.ffn_kernel_size > 1 {
            ((config.ffn_kernel_size - 1) / 2, config.ffn_kernel_size / 2)
        } else {
            (0, 0)
        };

        Ok(Self {
            conv_1,
            conv_2,
            pad_left,
            pad_right,
            hidden_act: config.hidden_act.clone(),
        })
    }

    fn act(&self, x: &Tensor) -> Result<Tensor> {
        match self.hidden_act.as_str() {
            "relu" => x.relu(),
            "gelu" => x.gelu_erf(),
            "tanh" => x.tanh(),
            "silu" => candle_nn::ops::silu(x),
            other => candle_core::bail!("unsupported hidden_act: {other}"),
        }
    }

    /// `hidden_states`: `(batch, seq_len, hidden_size)`.
    /// `padding_mask`: `(batch, seq_len, 1)` (channel-last, matches caller).
    pub fn forward(&self, hidden_states: &Tensor, padding_mask: &Tensor) -> Result<Tensor> {
        // permute(0, 2, 1): (batch, seq_len, C) -> (batch, C, seq_len)
        let mut hidden_states = hidden_states.transpose(1, 2)?.contiguous()?;
        let padding_mask = padding_mask.transpose(1, 2)?.contiguous()?;

        hidden_states = hidden_states.broadcast_mul(&padding_mask)?;
        hidden_states = pad_last_dim(&hidden_states, self.pad_left, self.pad_right)?;

        hidden_states = self.conv_1.forward(&hidden_states)?;
        hidden_states = self.act(&hidden_states)?;

        hidden_states = hidden_states.broadcast_mul(&padding_mask)?;
        hidden_states = pad_last_dim(&hidden_states, self.pad_left, self.pad_right)?;

        hidden_states = self.conv_2.forward(&hidden_states)?;
        hidden_states = hidden_states.broadcast_mul(&padding_mask)?;

        // permute back to (batch, seq_len, C)
        hidden_states.transpose(1, 2)?.contiguous()
    }
}

/// Pads the last dim (time axis, since tensor is channel-first here) with
/// zeros on both sides (mirrors `nn.functional.pad(x, [left, right, 0, 0, 0, 0])`).
fn pad_last_dim(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let dim = x.rank() - 1;
    let mut parts: Vec<Tensor> = Vec::new();
    if left > 0 {
        let mut shape = x.dims().to_vec();
        shape[dim] = left;
        parts.push(Tensor::zeros(shape.as_slice(), x.dtype(), x.device())?);
    }
    parts.push(x.clone());
    if right > 0 {
        let mut shape = x.dims().to_vec();
        shape[dim] = right;
        parts.push(Tensor::zeros(shape.as_slice(), x.dtype(), x.device())?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, dim)
}

// ---------------------------------------------------------------------
// Encoder layer (self-attention + FFN, pre-LN residual style)
// ---------------------------------------------------------------------

pub struct VitsEncoderLayer {
    attention: VitsAttention,
    layer_norm: LayerNorm,
    feed_forward: VitsFeedForward,
    final_layer_norm: LayerNorm,
}

impl VitsEncoderLayer {
    pub fn new(config: &VitsConfig, vb: VarBuilder) -> Result<Self> {
        let attention = VitsAttention::new(config, vb.pp("attention"))?;
        let layer_norm =
            candle_nn::layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("layer_norm"))?;
        let feed_forward = VitsFeedForward::new(config, vb.pp("feed_forward"))?;
        let final_layer_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("final_layer_norm"),
        )?;

        Ok(Self {
            attention,
            layer_norm,
            feed_forward,
            final_layer_norm,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        padding_mask: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let mut hidden_states = self.attention.forward(hidden_states, attention_mask)?;
        hidden_states = self.layer_norm.forward(&residual.add(&hidden_states)?)?;

        let residual = hidden_states.clone();
        let mut hidden_states = self.feed_forward.forward(&hidden_states, padding_mask)?;
        hidden_states = self.final_layer_norm.forward(&residual.add(&hidden_states)?)?;

        Ok(hidden_states)
    }
}

// ---------------------------------------------------------------------
// Encoder (stack of layers)
// ---------------------------------------------------------------------

pub struct VitsEncoder {
    layers: Vec<VitsEncoderLayer>,
}

impl VitsEncoder {
    pub fn new(config: &VitsConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(VitsEncoderLayer::new(config, vb.pp(format!("layers.{i}")))?);
        }
        Ok(Self { layers })
    }

    /// `hidden_states`: `(batch, seq_len, hidden_size)`.
    /// `padding_mask`: `(batch, seq_len, 1)`.
    /// `attention_mask`: additive `(batch, 1, seq_len, seq_len)`, built by
    /// the caller via [`build_bidirectional_mask`] (or `None` for an
    /// unpadded, single-sequence batch, which is what the CLI always uses).
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        padding_mask: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden_states = hidden_states.broadcast_mul(padding_mask)?;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, padding_mask, attention_mask)?;
        }
        hidden_states.broadcast_mul(padding_mask)
    }
}

/// Minimal replacement for the Python `create_bidirectional_mask`: builds
/// an additive float mask of shape `(batch, 1, seq_len, seq_len)` from a
/// `(batch, seq_len)` 0/1 attention mask. Returns `None` when
/// `attention_mask` is `None` (matches Python).
pub fn build_bidirectional_mask(attention_mask: Option<&Tensor>) -> Result<Option<Tensor>> {
    let Some(attention_mask) = attention_mask else {
        return Ok(None);
    };
    let (batch_size, seq_len) = attention_mask.dims2()?;
    let expanded = attention_mask
        .reshape((batch_size, 1, 1, seq_len))?
        .to_dtype(candle_core::DType::F32)?;
    let inverted = (1.0 - expanded)?;
    // masked_fill(inverted.bool(), finfo.min): wherever inverted != 0, use a
    // large negative number; elsewhere keep 0.
    let neg_inf = Tensor::full(f32::MIN, inverted.shape(), inverted.device())?;
    let zeros = Tensor::zeros_like(&inverted)?;
    let is_masked = inverted.ne(0f32)?;
    let additive = is_masked.where_cond(&neg_inf, &zeros)?;
    let additive = additive.broadcast_as((batch_size, 1, seq_len, seq_len))?;
    Ok(Some(additive))
}

#[cfg(test)]
mod tests {
    use super::build_bidirectional_mask;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn bidirectional_mask_is_f32_and_square() {
        let device = Device::Cpu;
        let mask = Tensor::from_vec(vec![1i64, 0, 1, 1, 1, 0], (2, 3), &device).unwrap();

        let mask = build_bidirectional_mask(Some(&mask)).unwrap().unwrap();

        assert_eq!(mask.dtype(), DType::F32);
        assert_eq!(mask.dims4().unwrap(), (2, 1, 3, 3));
    }
}

// ---------------------------------------------------------------------
// Text encoder (embedding + encoder + projection to prior mean/log-var)
// ---------------------------------------------------------------------

pub struct VitsTextEncoder {
    embed_tokens: Embedding,
    encoder: VitsEncoder,
    project: Conv1d,
    hidden_size: usize,
    flow_size: usize,
}

pub struct VitsTextEncoderOutput {
    pub last_hidden_state: Tensor,
    pub prior_means: Tensor,
    pub prior_log_variances: Tensor,
}

impl VitsTextEncoder {
    pub fn new(config: &VitsConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let encoder = VitsEncoder::new(config, vb.pp("encoder"))?;
        let project = candle_nn::conv1d(
            config.hidden_size,
            config.flow_size * 2,
            1,
            Conv1dConfig::default(),
            vb.pp("project"),
        )?;

        Ok(Self {
            embed_tokens,
            encoder,
            project,
            hidden_size: config.hidden_size,
            flow_size: config.flow_size,
        })
    }

    /// `input_ids`: `(batch, seq_len)` (i64/u32 token ids).
    /// `padding_mask`: `(batch, seq_len, 1)`.
    /// `attention_mask`: raw `(batch, seq_len)` 0/1 mask, or `None`.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        padding_mask: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<VitsTextEncoderOutput> {
        let hidden_states = (self.embed_tokens.forward(input_ids)? * (self.hidden_size as f64).sqrt())?;

        let bidirectional_mask = build_bidirectional_mask(attention_mask)?;
        let last_hidden_state =
            self.encoder
                .forward(&hidden_states, padding_mask, bidirectional_mask.as_ref())?;

        let stats = self
            .project
            .forward(&last_hidden_state.transpose(1, 2)?.contiguous()?)?
            .transpose(1, 2)?
            .contiguous()?
            .broadcast_mul(padding_mask)?;

        let prior_means = stats.narrow(2, 0, self.flow_size)?;
        let prior_log_variances = stats.narrow(2, self.flow_size, self.flow_size)?;

        Ok(VitsTextEncoderOutput {
            last_hidden_state,
            prior_means,
            prior_log_variances,
        })
    }
}
