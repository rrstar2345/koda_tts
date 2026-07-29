//! Port of `VitsHifiGan` and `HifiGanResidualBlock` from `model.py`.
//!
//! `apply_weight_norm`/`remove_weight_norm` (export/training utilities,
//! never called during inference) are not ported (see CONTEXT.md).

use super::wavenet::load_weight_norm_conv1d;
use crate::config::VitsConfig;
use candle_core::{Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};

fn leaky_relu(x: &Tensor, slope: f64) -> Result<Tensor> {
    let zeros = Tensor::zeros_like(x)?;
    let pos = x.maximum(&zeros)?;
    let neg = x.minimum(&zeros)?;
    pos.add(&(neg * slope)?)
}

struct HifiGanResidualBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    leaky_relu_slope: f64,
}

impl HifiGanResidualBlock {
    fn new(
        channels: usize,
        kernel_size: usize,
        dilation: &[usize],
        leaky_relu_slope: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let get_padding = |k: usize, d: usize| (k * d - d) / 2;

        let mut convs1 = Vec::with_capacity(dilation.len());
        let mut convs2 = Vec::with_capacity(dilation.len());
        for (i, &d) in dilation.iter().enumerate() {
            let cfg1 = Conv1dConfig {
                padding: get_padding(kernel_size, d),
                stride: 1,
                dilation: d,
                groups: 1,
                cudnn_fwd_algo: None,
            };
            convs1.push(load_weight_norm_conv1d(
                &vb.pp(format!("convs1.{i}")),
                channels,
                channels,
                kernel_size,
                cfg1,
                true,
            )?);

            let cfg2 = Conv1dConfig {
                padding: get_padding(kernel_size, 1),
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            };
            convs2.push(load_weight_norm_conv1d(
                &vb.pp(format!("convs2.{i}")),
                channels,
                channels,
                kernel_size,
                cfg2,
                true,
            )?);
        }

        Ok(Self {
            convs1,
            convs2,
            leaky_relu_slope,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for (conv1, conv2) in self.convs1.iter().zip(self.convs2.iter()) {
            let residual = hidden_states.clone();
            hidden_states = leaky_relu(&hidden_states, self.leaky_relu_slope)?;
            hidden_states = conv1.forward(&hidden_states)?;
            hidden_states = leaky_relu(&hidden_states, self.leaky_relu_slope)?;
            hidden_states = conv2.forward(&hidden_states)?;
            hidden_states = hidden_states.add(&residual)?;
        }
        Ok(hidden_states)
    }
}

pub struct VitsHifiGan {
    num_kernels: usize,
    num_upsamples: usize,
    conv_pre: Conv1d,
    upsampler: Vec<ConvTranspose1d>,
    resblocks: Vec<HifiGanResidualBlock>,
    conv_post: Conv1d,
    cond: Option<Conv1d>,
    leaky_relu_slope: f64,
}

impl VitsHifiGan {
    pub fn new(config: &VitsConfig, vb: VarBuilder) -> Result<Self> {
        let num_kernels = config.resblock_kernel_sizes.len();
        let num_upsamples = config.upsample_rates.len();

        let conv_pre_cfg = Conv1dConfig {
            padding: 3,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv_pre = candle_nn::conv1d(
            config.flow_size,
            config.upsample_initial_channel,
            7,
            conv_pre_cfg,
            vb.pp("conv_pre"),
        )?;

        let mut upsampler = Vec::with_capacity(num_upsamples);
        for (i, (&upsample_rate, &kernel_size)) in config
            .upsample_rates
            .iter()
            .zip(config.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let in_ch = config.upsample_initial_channel / 2usize.pow(i as u32);
            let out_ch = config.upsample_initial_channel / 2usize.pow((i + 1) as u32);
            let padding = (kernel_size - upsample_rate) / 2;

            let cfg = ConvTranspose1dConfig {
                padding,
                output_padding: 0,
                stride: upsample_rate,
                dilation: 1,
                groups: 1,
            };
            let layer_vb = vb.pp(format!("upsampler.{i}"));
            let layer = load_weight_norm_conv_transpose1d(&layer_vb, in_ch, out_ch, kernel_size, cfg)?;
            upsampler.push(layer);
        }

        let mut resblocks = Vec::with_capacity(num_upsamples * num_kernels);
        let mut last_channels = config.upsample_initial_channel;
        for i in 0..num_upsamples {
            let channels = config.upsample_initial_channel / 2usize.pow((i + 1) as u32);
            last_channels = channels;
            for (j, (&kernel_size, dilation)) in config
                .resblock_kernel_sizes
                .iter()
                .zip(config.resblock_dilation_sizes.iter())
                .enumerate()
            {
                let idx = i * num_kernels + j;
                resblocks.push(HifiGanResidualBlock::new(
                    channels,
                    kernel_size,
                    dilation,
                    config.leaky_relu_slope,
                    vb.pp(format!("resblocks.{idx}")),
                )?);
            }
        }

        let conv_post_cfg = Conv1dConfig {
            padding: 3,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv_post_vb = vb.pp("conv_post");
        let conv_post = Conv1d::new(
            conv_post_vb.get((1, last_channels, 7), "weight")?,
            None, // bias=False in Python
            conv_post_cfg,
        );

        let cond = if config.speaker_embedding_size != 0 {
            Some(candle_nn::conv1d(
                config.speaker_embedding_size,
                config.upsample_initial_channel,
                1,
                Conv1dConfig::default(),
                vb.pp("cond"),
            )?)
        } else {
            None
        };

        Ok(Self {
            num_kernels,
            num_upsamples,
            conv_pre,
            upsampler,
            resblocks,
            conv_post,
            cond,
            leaky_relu_slope: config.leaky_relu_slope,
        })
    }

    /// Converts a spectrogram `(batch, spectrogram_bins/flow_size, seq_len)`
    /// into a waveform `(batch, 1, num_frames)`.
    pub fn forward(&self, spectrogram: &Tensor, global_conditioning: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = self.conv_pre.forward(spectrogram)?;

        if let (Some(gc), Some(cond)) = (global_conditioning, &self.cond) {
            hidden_states = hidden_states.add(&cond.forward(gc)?)?;
        }

        for i in 0..self.num_upsamples {
            hidden_states = leaky_relu(&hidden_states, self.leaky_relu_slope)?;
            hidden_states = self.upsampler[i].forward(&hidden_states)?;

            let mut res_state = self.resblocks[i * self.num_kernels].forward(&hidden_states)?;
            for j in 1..self.num_kernels {
                res_state = res_state.add(&self.resblocks[i * self.num_kernels + j].forward(&hidden_states)?)?;
            }
            hidden_states = (res_state / self.num_kernels as f64)?;
        }

        hidden_states = leaky_relu(&hidden_states, 0.01)?; // default nn.functional.leaky_relu slope
        hidden_states = self.conv_post.forward(&hidden_states)?;
        hidden_states.tanh()
    }
}

/// Loads a `ConvTranspose1d` whose weights were saved by PyTorch's
/// `weight_norm` parametrization (see `wavenet::load_weight_norm_conv1d`
/// for the naming-scheme notes; identical logic, transposed-conv weight
/// shape `(in_channels, out_channels, kernel_size)`).
fn load_weight_norm_conv_transpose1d(
    vb: &VarBuilder,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: ConvTranspose1dConfig,
) -> Result<ConvTranspose1d> {
    let shape = (in_channels, out_channels, kernel_size);

    let (g, v) = if vb.contains_tensor("weight_g") && vb.contains_tensor("weight_v") {
        (vb.get((in_channels, 1, 1), "weight_g")?, vb.get(shape, "weight_v")?)
    } else {
        (
            vb.get((in_channels, 1, 1), "parametrizations.weight.original0")?,
            vb.get(shape, "parametrizations.weight.original1")?,
        )
    };

    let v_sq_sum = v.sqr()?.sum_keepdim((1, 2))?;
    let v_norm = v_sq_sum.sqrt()?;
    let weight = v.broadcast_div(&v_norm)?.broadcast_mul(&g)?;

    let bias = vb.get(out_channels, "bias")?;

    Ok(ConvTranspose1d::new(weight, Some(bias), cfg))
}