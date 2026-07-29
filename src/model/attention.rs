//! Port of `VitsAttention` (multi-headed attention with relative
//! positional representation) from `model.py`.
//!
//! `output_attentions` plumbing is dropped (see CONTEXT.md): this port
//! always returns just the attention output.

use crate::config::VitsConfig;
use candle_core::{DType, Result, Tensor, D, Module};
use candle_nn::{Linear, VarBuilder};

pub struct VitsAttention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    scaling: f64,
    window_size: usize,
    k_proj: Linear,
    v_proj: Linear,
    q_proj: Linear,
    out_proj: Linear,
    emb_rel_k: Tensor,
    emb_rel_v: Tensor,
}

impl VitsAttention {
    pub fn new(config: &VitsConfig, vb: VarBuilder) -> Result<Self> {
        let embed_dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let window_size = config.window_size;
        let head_dim = embed_dim / num_heads;
        if head_dim * num_heads != embed_dim {
            candle_core::bail!(
                "hidden_size must be divisible by num_attention_heads (got hidden_size: {} \
                 and num_attention_heads: {})",
                embed_dim,
                num_heads
            );
        }
        let scaling = (head_dim as f64).powf(-0.5);

        let k_proj = linear(&vb.pp("k_proj"), embed_dim, embed_dim, config.use_bias)?;
        let v_proj = linear(&vb.pp("v_proj"), embed_dim, embed_dim, config.use_bias)?;
        let q_proj = linear(&vb.pp("q_proj"), embed_dim, embed_dim, config.use_bias)?;
        let out_proj = linear(&vb.pp("out_proj"), embed_dim, embed_dim, config.use_bias)?;

        let emb_rel_k = vb.get((1, window_size * 2 + 1, head_dim), "emb_rel_k")?;
        let emb_rel_v = vb.get((1, window_size * 2 + 1, head_dim), "emb_rel_v")?;

        Ok(Self {
            embed_dim,
            num_heads,
            head_dim,
            scaling,
            window_size,
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            emb_rel_k,
            emb_rel_v,
        })
    }

    fn shape(&self, tensor: &Tensor, seq_len: usize, bsz: usize) -> Result<Tensor> {
        tensor
            .reshape((bsz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// `attention_mask`, if given, has shape `(bsz, 1, tgt_len, src_len)`
    /// and is additive (0 for keep, large-negative for mask-out).
    pub fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (bsz, tgt_len, _) = hidden_states.dims3()?;

        let query_states = (self.q_proj.forward(hidden_states)? * self.scaling)?;

        let key_states = self.shape(&self.k_proj.forward(hidden_states)?, tgt_len, bsz)?;
        let value_states = self.shape(&self.v_proj.forward(hidden_states)?, tgt_len, bsz)?;

        let proj_bsz = bsz * self.num_heads;
        let query_states = self
            .shape(&query_states, tgt_len, bsz)?
            .reshape((proj_bsz, tgt_len, self.head_dim))?;
        let key_states = key_states.reshape((proj_bsz, tgt_len, self.head_dim))?;
        let value_states = value_states.reshape((proj_bsz, tgt_len, self.head_dim))?;

        let src_len = tgt_len;
        let mut attn_weights = query_states.matmul(&key_states.transpose(1, 2)?)?;

        // Relative positional bias (window_size is always set for this model).
        let key_relative_embeddings = self.get_relative_embeddings(&self.emb_rel_k, src_len)?;
        let key_relative_embeddings = key_relative_embeddings.broadcast_as((
            proj_bsz,
            key_relative_embeddings.dim(1)?,
            self.head_dim,
        ))?;
        let relative_logits =
            query_states.matmul(&key_relative_embeddings.transpose(D::Minus2, D::Minus1)?)?;
        let rel_pos_bias = self.relative_position_to_absolute_position(&relative_logits)?;
        attn_weights = attn_weights.add(&rel_pos_bias)?;

        if let Some(mask) = attention_mask {
            let attn_weights_4d = attn_weights.reshape((bsz, self.num_heads, tgt_len, src_len))?;
            attn_weights = attn_weights_4d
                .broadcast_add(mask)?
                .reshape((proj_bsz, tgt_len, src_len))?;
        }

        attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let attn_probs = attn_weights.clone();

        let mut attn_output = attn_probs.matmul(&value_states)?;

        let value_relative_embeddings = self.get_relative_embeddings(&self.emb_rel_v, src_len)?;
        let relative_weights = self.absolute_position_to_relative_position(&attn_probs)?;
        let value_relative_embeddings = value_relative_embeddings.broadcast_as((
            proj_bsz,
            value_relative_embeddings.dim(1)?,
            self.head_dim,
        ))?;
        let rel_pos_bias_v = relative_weights.matmul(&value_relative_embeddings)?;
        attn_output = attn_output.add(&rel_pos_bias_v)?;

        let attn_output = attn_output
            .reshape((bsz, self.num_heads, tgt_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((bsz, tgt_len, self.embed_dim))?
            .contiguous()?;

        self.out_proj.forward(&attn_output)
    }

    fn get_relative_embeddings(&self, relative_embeddings: &Tensor, length: usize) -> Result<Tensor> {
        let window = self.window_size;
        let pad_length = length.saturating_sub(window + 1);
        let padded = if pad_length > 0 {
            pad_dim1(relative_embeddings, pad_length, pad_length)?
        } else {
            relative_embeddings.clone()
        };

        let slice_start = (window + 1).saturating_sub(length);
        let slice_len = 2 * length - 1;
        padded.narrow(1, slice_start, slice_len)
    }

    /// Converts relative-position logits of shape `(batch_heads, length, 2*length-1)`
    /// to absolute-position logits of shape `(batch_heads, length, length)`.
    fn relative_position_to_absolute_position(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_heads, length, _) = x.dims3()?;

        let x = pad_last_dim_right(x, 1)?;
        let x_flat = x.reshape((batch_heads, length * 2 * length))?;
        let x_flat = pad_last_dim_right(&x_flat, length - 1)?;

        let x_final = x_flat.reshape((batch_heads, length + 1, 2 * length - 1))?;
        x_final.narrow(1, 0, length)?.narrow(2, length - 1, length)
    }

    /// Converts absolute-position attention probs of shape
    /// `(batch_heads, length, length)` to relative-position weights of
    /// shape `(batch_heads, length, 2*length-1)`.
    fn absolute_position_to_relative_position(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_heads, length, _) = x.dims3()?;

        let x = pad_last_dim_right(x, length - 1)?;
        let x_flat = x.reshape((batch_heads, length * (2 * length - 1)))?;
        let x_flat = pad_first_dim_of_flat(&x_flat, length)?;
        let x_final = x_flat.reshape((batch_heads, length, 2 * length))?;
        x_final.narrow(2, 1, 2 * length - 1)
    }
}

fn linear(vb: &VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> Result<Linear> {
    if bias {
        candle_nn::linear(in_dim, out_dim, vb.clone())
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb.clone())
    }
}

/// Pads dim 1 (the middle dim of a 3D tensor) with `before`/`after` zeros.
fn pad_dim1(x: &Tensor, before: usize, after: usize) -> Result<Tensor> {
    let dims = x.dims3()?;
    let mut parts: Vec<Tensor> = Vec::new();
    if before > 0 {
        parts.push(Tensor::zeros((dims.0, before, dims.2), x.dtype(), x.device())?);
    }
    parts.push(x.clone());
    if after > 0 {
        parts.push(Tensor::zeros((dims.0, after, dims.2), x.dtype(), x.device())?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 1)
}

/// Pads the last dimension of a 3D tensor with `n` zeros on the right.
fn pad_last_dim_right(x: &Tensor, n: usize) -> Result<Tensor> {
    if n == 0 {
        return Ok(x.clone());
    }
    let mut dims = x.dims().to_vec();
    let last = dims.len() - 1;
    dims[last] = n;
    let zeros = Tensor::zeros(dims.as_slice(), x.dtype(), x.device())?;
    Tensor::cat(&[x, &zeros], last)
}

/// Pads a flat (2D) tensor with `n` zeros at the start of dim 1 (mirrors
/// `nn.functional.pad(x_flat, [length, 0])`).
fn pad_first_dim_of_flat(x: &Tensor, n: usize) -> Result<Tensor> {
    let (batch, _) = x.dims2()?;
    let zeros = Tensor::zeros((batch, n), x.dtype(), x.device())?;
    Tensor::cat(&[&zeros, x], 1)
}

#[allow(unused)]
fn _unused_dtype(d: DType) -> DType {
    d
}
