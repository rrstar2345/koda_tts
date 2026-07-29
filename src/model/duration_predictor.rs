//! Port of `VitsStochasticDurationPredictor` from `model.py`.
//!
//! The non-stochastic `VitsDurationPredictor` variant from the Python
//! source is **not** ported: this checkpoint's `config.json` always sets
//! `use_stochastic_duration_prediction=true` (see CONTEXT.md), so only this
//! path is reachable. `VitsModel::new` asserts the config flag at load time.
//!
//! Only the `reverse=true` (inference) branch is exercised by
//! `VitsModel::forward`; the `reverse=false` branch computes a training
//! loss (`nll + logq`) that's never used for inference and is not ported.

use super::flows::{VitsConvFlow, VitsDilatedDepthSeparableConv, VitsElementwiseAffine};
use super::tensor_ext::FlipExt;
use crate::config::VitsConfig;
use candle_core::{Device, Result, Tensor, Module};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};

enum Flow {
    Affine(VitsElementwiseAffine),
    Conv(VitsConvFlow),
}

impl Flow {
    fn forward(
        &self,
        inputs: &Tensor,
        padding_mask: &Tensor,
        global_conditioning: Option<&Tensor>,
        reverse: bool,
    ) -> Result<Tensor> {
        match self {
            Flow::Affine(f) => Ok(f.forward(inputs, padding_mask, reverse)?.0),
            Flow::Conv(f) => Ok(f.forward(inputs, padding_mask, global_conditioning, reverse)?.0),
        }
    }
}

pub struct VitsStochasticDurationPredictor {
    conv_pre: Conv1d,
    conv_proj: Conv1d,
    conv_dds: VitsDilatedDepthSeparableConv,
    cond: Option<Conv1d>,
    flows: Vec<Flow>,
    device: Device,
}

impl VitsStochasticDurationPredictor {
    pub fn new(config: &VitsConfig, vb: VarBuilder) -> Result<Self> {
        let embed_dim = config.speaker_embedding_size;
        let filter_channels = config.hidden_size;

        let conv_pre = conv1d(&vb.pp("conv_pre"), filter_channels, filter_channels, 1)?;
        let conv_proj = conv1d(&vb.pp("conv_proj"), filter_channels, filter_channels, 1)?;
        let conv_dds = VitsDilatedDepthSeparableConv::new(
            config,
            config.duration_predictor_dropout,
            vb.pp("conv_dds"),
        )?;

        let cond = if embed_dim != 0 {
            Some(conv1d(&vb.pp("cond"), embed_dim, filter_channels, 1)?)
        } else {
            None
        };

        // Only the reverse (inference) flows list is needed for
        // `forward(reverse=true)`: it uses `self.flows` (not `post_flows`,
        // which is training-only, feeding the posterior encoder path that
        // is unreachable at inference). We still load `self.flows`' weights
        // under the same variable names as Python for state_dict parity.
        let mut flows = Vec::with_capacity(1 + config.duration_predictor_num_flows);
        flows.push(Flow::Affine(VitsElementwiseAffine::new(config, vb.pp("flows.0"))?));
        for i in 0..config.duration_predictor_num_flows {
            flows.push(Flow::Conv(VitsConvFlow::new(config, vb.pp(format!("flows.{}", i + 1)))?));
        }

        Ok(Self {
            conv_pre,
            conv_proj,
            conv_dds,
            cond,
            flows,
            device: vb.device().clone(),
        })
    }

    /// Inference-only forward pass (`reverse=true` in Python). Returns the
    /// predicted `log_duration` tensor of shape `(batch, 1, seq_len)`.
    pub fn forward(
        &self,
        inputs: &Tensor,
        padding_mask: &Tensor,
        global_conditioning: Option<&Tensor>,
        noise_scale: f64,
    ) -> Result<Tensor> {
        let mut inputs = self.conv_pre.forward(inputs)?;

        if let (Some(gc), Some(cond)) = (global_conditioning, &self.cond) {
            inputs = inputs.add(&cond.forward(gc)?)?;
        }

        inputs = self.conv_dds.forward(&inputs, padding_mask, None)?;
        inputs = self.conv_proj.forward(&inputs)?.broadcast_mul(padding_mask)?;

        // Python: `flows = list(reversed(self.flows)); flows = flows[:-2] + [flows[-1]]`
        // i.e. reverse the list, then drop the second-to-last element
        // (the "useless vflow"). With our ordering [Affine, Conv, Conv, Conv, Conv]
        // (5 elements for the default `duration_predictor_num_flows=4`),
        // reversed is [Conv, Conv, Conv, Conv, Affine]; dropping index -2
        // (second-to-last = the 4th Conv) yields
        // [Conv, Conv, Conv, Affine].
        let reversed: Vec<&Flow> = self.flows.iter().rev().collect();
        let mut ordered: Vec<&Flow> = Vec::with_capacity(reversed.len() - 1);
        ordered.extend_from_slice(&reversed[..reversed.len() - 2]);
        ordered.push(reversed[reversed.len() - 1]);

        let (batch, _channels, seq_len) = inputs.dims3()?;
        let mut latents = (Tensor::randn(0f32, 1f32, (batch, 2, seq_len), &self.device)?
            * noise_scale)?;

        for flow in ordered {
            latents = latents.flip(&[1])?;
            latents = flow.forward(&latents, padding_mask, Some(&inputs), true)?;
        }

        let log_duration = latents.narrow(1, 0, 1)?;
        Ok(log_duration)
    }
}

fn conv1d(vb: &VarBuilder, in_channels: usize, out_channels: usize, kernel_size: usize) -> Result<Conv1d> {
    candle_nn::conv1d(in_channels, out_channels, kernel_size, Conv1dConfig::default(), vb.clone())
}
