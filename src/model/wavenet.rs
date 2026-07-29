//! Port of `VitsWaveNet` and `fused_add_tanh_sigmoid_multiply` from `model.py`.
//!
//! Speaker conditioning (`global_conditioning`) is kept as `Option<&Tensor>`
//! for architectural fidelity, but this checkpoint has
//! `speaker_embedding_size == 0`, so `cond_layer` is never constructed and
//! callers always pass `None` (see CONTEXT.md).

use crate::config::VitsConfig;
use candle_core::{Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};

pub(crate) fn fused_add_tanh_sigmoid_multiply(
    input_a: &Tensor,
    input_b: &Tensor,
    num_channels: usize,
) -> Result<Tensor> {
    let in_act = input_a.add(input_b)?;
    let t_act = in_act.narrow(1, 0, num_channels)?.tanh()?;
    let s_act_input = in_act.narrow(1, num_channels, in_act.dim(1)? - num_channels)?;
    // sigmoid(x) = 1 / (1 + exp(-x))
    let s_act = (s_act_input.neg()?.exp()? + 1.0)?.recip()?;
    t_act.mul(&s_act)
}

pub struct VitsWaveNet {
    hidden_size: usize,
    num_layers: usize,
    in_layers: Vec<Conv1d>,
    res_skip_layers: Vec<Conv1d>,
    cond_layer: Option<Conv1d>,
    dropout_p: f64,
}

impl VitsWaveNet {
    pub fn new(config: &VitsConfig, num_layers: usize, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;

        let cond_layer = if config.speaker_embedding_size != 0 {
            let cfg = Conv1dConfig::default();
            Some(Conv1d::new(
                vb.get(
                    (2 * hidden_size * num_layers, config.speaker_embedding_size, 1),
                    "cond_layer.weight",
                )?,
                Some(vb.get(2 * hidden_size * num_layers, "cond_layer.bias")?),
                cfg,
            ))
        } else {
            None
        };

        let mut in_layers = Vec::with_capacity(num_layers);
        let mut res_skip_layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let dilation = config.wavenet_dilation_rate.pow(i as u32);
            let padding = (config.wavenet_kernel_size * dilation - dilation) / 2;

            let in_cfg = Conv1dConfig {
                padding,
                stride: 1,
                dilation,
                groups: 1,
                cudnn_fwd_algo: None,
            };
            let in_layer_vb = vb.pp(format!("in_layers.{i}"));
            let in_layer = load_weight_norm_conv1d(
                &in_layer_vb,
                hidden_size,
                2 * hidden_size,
                config.wavenet_kernel_size,
                in_cfg,
                true,
            )?;
            in_layers.push(in_layer);

            let res_skip_channels = if i < num_layers - 1 {
                2 * hidden_size
            } else {
                hidden_size
            };
            let res_skip_vb = vb.pp(format!("res_skip_layers.{i}"));
            let res_skip_layer = load_weight_norm_conv1d(
                &res_skip_vb,
                hidden_size,
                res_skip_channels,
                1,
                Conv1dConfig::default(),
                true,
            )?;
            res_skip_layers.push(res_skip_layer);
        }

        Ok(Self {
            hidden_size,
            num_layers,
            in_layers,
            res_skip_layers,
            cond_layer,
            dropout_p: config.wavenet_dropout,
        })
    }

    pub fn forward(
        &self,
        inputs: &Tensor,
        padding_mask: &Tensor,
        global_conditioning: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut inputs = inputs.clone();
        let mut outputs = Tensor::zeros_like(&inputs)?;

        let global_conditioning = match (global_conditioning, &self.cond_layer) {
            (Some(gc), Some(cond_layer)) => Some(cond_layer.forward(gc)?),
            _ => None,
        };

        for i in 0..self.num_layers {
            let hidden_states = self.in_layers[i].forward(&inputs)?;

            let global_states = if let Some(gc) = &global_conditioning {
                let cond_offset = i * 2 * self.hidden_size;
                gc.narrow(1, cond_offset, 2 * self.hidden_size)?
            } else {
                Tensor::zeros_like(&hidden_states)?
            };

            let mut acts =
                fused_add_tanh_sigmoid_multiply(&hidden_states, &global_states, self.hidden_size)?;
            if self.dropout_p > 0.0 {
                acts = candle_nn::ops::dropout(&acts, self.dropout_p as f32)?;
            }

            let res_skip_acts = self.res_skip_layers[i].forward(&acts)?;
            if i < self.num_layers - 1 {
                let res_acts = res_skip_acts.narrow(1, 0, self.hidden_size)?;
                inputs = inputs.add(&res_acts)?.mul(padding_mask)?;
                let skip = res_skip_acts.narrow(
                    1,
                    self.hidden_size,
                    res_skip_acts.dim(1)? - self.hidden_size,
                )?;
                outputs = outputs.add(&skip)?;
            } else {
                outputs = outputs.add(&res_skip_acts)?;
            }
        }

        outputs.mul(padding_mask)
    }
}

/// Loads a `Conv1d` whose weights were saved by PyTorch's `weight_norm`
/// parametrization. Two naming schemes are supported (see CONTEXT.md):
/// - legacy `torch.nn.utils.weight_norm`: `<prefix>.weight_g`, `<prefix>.weight_v`
/// - new `torch.nn.utils.parametrizations.weight_norm`:
///   `<prefix>.parametrizations.weight.original0` (g) and `.original1` (v)
///
/// The reconstructed weight is `g * v / ||v||`, with the norm taken over all
/// dims except dim 0 (out_channels), matching PyTorch's default `dim=0`.
pub(crate) fn load_weight_norm_conv1d(
    vb: &VarBuilder,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv1dConfig,
    bias: bool,
) -> Result<Conv1d> {
    let shape = (out_channels, in_channels, kernel_size);

    let (g, v) = if vb.contains_tensor("weight_g") && vb.contains_tensor("weight_v") {
        (vb.get((out_channels, 1, 1), "weight_g")?, vb.get(shape, "weight_v")?)
    } else {
        (
            vb.get((out_channels, 1, 1), "parametrizations.weight.original0")?,
            vb.get(shape, "parametrizations.weight.original1")?,
        )
    };

    // norm over (in_channels, kernel_size) per output channel
    let v_sq_sum = v.sqr()?.sum_keepdim((1, 2))?;
    let v_norm = v_sq_sum.sqrt()?;
    let weight = v.broadcast_div(&v_norm)?.broadcast_mul(&g)?;

    let bias_tensor = if bias {
        Some(vb.get(out_channels, "bias")?)
    } else {
        None
    };

    Ok(Conv1d::new(weight, bias_tensor, cfg))
}