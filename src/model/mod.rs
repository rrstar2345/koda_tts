//! Port of the top-level `VitsModel` from `model.py`.
//!
//! Multi-speaker branches, `output_attentions`/`output_hidden_states`,
//! `labels`/training, and `VitsPosteriorEncoder` (training-only) are all
//! dropped — see CONTEXT.md. This forward always runs the
//! `use_stochastic_duration_prediction=true`, `reverse=true` (inference)
//! path for a single-speaker checkpoint.

pub mod attention;
pub mod duration_predictor;
pub mod flows;
pub mod hifigan;
pub mod tensor_ext;
pub mod text_encoder;
pub mod wavenet;

use crate::config::VitsConfig;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use duration_predictor::VitsStochasticDurationPredictor;
use flows::VitsResidualCouplingBlock;
use hifigan::VitsHifiGan;
use text_encoder::VitsTextEncoder;

pub struct VitsModelOutput {
    /// `(batch, num_samples)`
    pub waveform: Tensor,
    /// `(batch,)`, in samples.
    pub sequence_lengths: Vec<i64>,
    /// `(batch, spectrogram_bins/flow_size, seq_len)`
    pub spectrogram: Tensor,
}

pub struct VitsModel {
    config: VitsConfig,
    text_encoder: VitsTextEncoder,
    flow: VitsResidualCouplingBlock,
    decoder: VitsHifiGan,
    duration_predictor: VitsStochasticDurationPredictor,
    speaking_rate: f64,
    noise_scale: f64,
    noise_scale_duration: f64,
    device: Device,
}

impl VitsModel {
    pub fn new(config: &VitsConfig, vb: VarBuilder) -> Result<Self> {
        if !config.use_stochastic_duration_prediction {
            candle_core::bail!(
                "this port only implements VitsStochasticDurationPredictor; \
                 config.use_stochastic_duration_prediction must be true"
            );
        }
        if config.num_speakers > 1 {
            candle_core::bail!(
                "this port only supports single-speaker checkpoints (config.num_speakers=1)"
            );
        }

        let text_encoder = VitsTextEncoder::new(config, vb.pp("text_encoder"))?;
        let flow = VitsResidualCouplingBlock::new(config, vb.pp("flow"))?;
        let decoder = VitsHifiGan::new(config, vb.pp("decoder"))?;
        let duration_predictor =
            VitsStochasticDurationPredictor::new(config, vb.pp("duration_predictor"))?;

        Ok(Self {
            config: config.clone(),
            text_encoder,
            flow,
            decoder,
            duration_predictor,
            speaking_rate: config.speaking_rate,
            noise_scale: config.noise_scale,
            noise_scale_duration: config.noise_scale_duration,
            device: vb.device().clone(),
        })
    }

    /// `input_ids`: `(batch, seq_len)` (u32 token ids).
    /// `attention_mask`: `(batch, seq_len)` 0/1 mask, or `None` (treated as
    /// all-ones, matching Python when `attention_mask` isn't passed).
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        speaking_rate: Option<f64>,
    ) -> Result<VitsModelOutput> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        let input_padding_mask = match attention_mask {
            Some(mask) => mask.to_dtype(DType::F32)?.unsqueeze(2)?,
            None => Tensor::ones((batch_size, seq_len, 1), DType::F32, &self.device)?,
        };

        let text_encoder_output =
            self.text_encoder
                .forward(input_ids, &input_padding_mask, attention_mask)?;

        // (batch, seq_len, C) -> (batch, C, seq_len)
        let hidden_states = text_encoder_output.last_hidden_state.transpose(1, 2)?.contiguous()?;
        let input_padding_mask = input_padding_mask.transpose(1, 2)?.contiguous()?;
        let prior_means = text_encoder_output.prior_means;
        let prior_log_variances = text_encoder_output.prior_log_variances;

        let log_duration = self.duration_predictor.forward(
            &hidden_states,
            &input_padding_mask,
            None,
            self.noise_scale_duration,
        )?;

        let speaking_rate = speaking_rate.unwrap_or(self.speaking_rate);
        let length_scale = 1.0 / speaking_rate;
        // duration = ceil(exp(log_duration) * input_padding_mask * length_scale)
        let duration = (log_duration.exp()?.broadcast_mul(&input_padding_mask)? * length_scale)?.ceil()?;

        // predicted_lengths = clamp_min(sum(duration, dims=[1,2]), 1)
        let predicted_lengths_f = duration.sum((1, 2))?; // (batch,)
        let predicted_lengths_vec: Vec<f32> = predicted_lengths_f.to_vec1()?;
        let predicted_lengths: Vec<i64> = predicted_lengths_vec
            .iter()
            .map(|&v| (v.max(1.0)) as i64)
            .collect();
        let max_output_length = *predicted_lengths.iter().max().unwrap_or(&0) as usize;

        // output_padding_mask: (batch, 1, max_output_length), 1 where index < predicted_length
        let mut output_mask_data = vec![0f32; batch_size * max_output_length];
        for b in 0..batch_size {
            let len = predicted_lengths[b] as usize;
            for t in 0..len.min(max_output_length) {
                output_mask_data[b * max_output_length + t] = 1.0;
            }
        }
        let output_padding_mask =
            Tensor::from_vec(output_mask_data, (batch_size, 1, max_output_length), &self.device)?;

        // attn_mask = unsqueeze(input_padding_mask, 2) * unsqueeze(output_padding_mask, -1)
        // input_padding_mask: (batch, 1, seq_len) -> unsqueeze(2) -> (batch, 1, 1, seq_len)
        // output_padding_mask: (batch, 1, out_len) -> unsqueeze(-1) -> (batch, 1, out_len, 1)
        let input_padding_mask_u = input_padding_mask.unsqueeze(2)?; // (batch, 1, 1, seq_len)
        let output_padding_mask_u = output_padding_mask.unsqueeze(3)?; // (batch, 1, out_len, 1)
        let attn_mask = input_padding_mask_u.broadcast_mul(&output_padding_mask_u)?; // (batch, 1, out_len, seq_len)

        // cum_duration: (batch, seq_len) — cumulative sum along seq_len (last dim of duration
        // which is (batch, 1, seq_len)).
        let duration_2d = duration.reshape((batch_size, seq_len))?;
        let cum_duration = duration_2d.cumsum(1)?; // (batch, seq_len)

        // valid_indices[b, i, t] = t < cum_duration[b, i]   (t in 0..output_length)
        let cum_duration_vec: Vec<f32> = cum_duration.flatten_all()?.to_vec1()?;
        let mut valid_indices_data = vec![0f32; batch_size * seq_len * max_output_length];
        for b in 0..batch_size {
            for i in 0..seq_len {
                let cd = cum_duration_vec[b * seq_len + i];
                for t in 0..max_output_length {
                    if (t as f32) < cd {
                        valid_indices_data[(b * seq_len + i) * max_output_length + t] = 1.0;
                    }
                }
            }
        }
        let valid_indices = Tensor::from_vec(
            valid_indices_data,
            (batch_size, seq_len, max_output_length),
            &self.device,
        )?;

        // padded_indices = valid_indices - pad(valid_indices, [0,0,1,0,0,0])[:, :-1]
        // i.e. shift valid_indices by +1 along the seq_len (input_length) dim,
        // inserting a zero row at index 0 and dropping the last row.
        let shifted = shift_dim1_down_by_one(&valid_indices)?;
        let padded_indices = (valid_indices - shifted)?; // (batch, seq_len, out_len)

        // attn = padded_indices.unsqueeze(1).transpose(2,3) * attn_mask
        // padded_indices.unsqueeze(1): (batch, 1, seq_len, out_len)
        // .transpose(2,3): (batch, 1, out_len, seq_len)
        let attn = padded_indices
            .unsqueeze(1)?
            .transpose(2, 3)?
            .contiguous()?
            .broadcast_mul(&attn_mask)?; // (batch, 1, out_len, seq_len)

        // prior_means = matmul(attn.squeeze(1), prior_means).transpose(1,2)
        let attn_sq = attn.squeeze(1)?; // (batch, out_len, seq_len)
        let prior_means = attn_sq.matmul(&prior_means)?.transpose(1, 2)?.contiguous()?; // (batch, flow_size, out_len)
        let prior_log_variances = attn_sq
            .matmul(&prior_log_variances)?
            .transpose(1, 2)?
            .contiguous()?;

        let noise = Tensor::randn(0f32, 1f32, prior_means.shape(), &self.device)?;
        let prior_latents = (prior_means
            + (noise.mul(&prior_log_variances.exp()?)? * self.noise_scale)?)?;

        let latents = self.flow.forward(&prior_latents, &output_padding_mask, None, true)?;

        let spectrogram = latents.broadcast_mul(&output_padding_mask)?;
        let waveform = self.decoder.forward(&spectrogram, None)?;
        let waveform = waveform.squeeze(1)?; // (batch, num_samples)

        let upsample_prod: i64 = self.config.upsample_rates.iter().map(|&r| r as i64).product();
        let sequence_lengths: Vec<i64> = predicted_lengths.iter().map(|&l| l * upsample_prod).collect();

        Ok(VitsModelOutput {
            waveform,
            sequence_lengths,
            spectrogram,
        })
    }
}

/// Mirrors `x - nn.functional.pad(x, [0,0,1,0,0,0])[:, :-1]` for a 3D
/// tensor `x` of shape `(batch, dim1, dim2)`: shifts along `dim1` by one
/// (zero-filled at index 0, dropping the last row).
fn shift_dim1_down_by_one(x: &Tensor) -> Result<Tensor> {
    let (batch, dim1, dim2) = x.dims3()?;
    let zero_row = Tensor::zeros((batch, 1, dim2), x.dtype(), x.device())?;
    let truncated = x.narrow(1, 0, dim1 - 1)?;
    Tensor::cat(&[&zero_row, &truncated], 1)
}