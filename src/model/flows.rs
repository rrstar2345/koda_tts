//! Port of the normalizing-flow components from `model.py`:
//! `VitsResidualCouplingLayer`, `VitsResidualCouplingBlock`, `VitsConvFlow`,
//! `VitsElementwiseAffine`, `VitsDilatedDepthSeparableConv`, and the
//! rational-quadratic spline functions used by `VitsConvFlow`.
//!
//! Only the `reverse=true` path is exercised by inference
//! (`VitsModel::forward` always calls flows with `reverse=true`), but both
//! directions are ported for architectural completeness and because the
//! stochastic duration predictor's training-time branch (`reverse=false`)
//! is also unused at inference and could be trimmed later if desired.
//! We keep both here since the code is shared/cheap and matches the source
//! 1:1, aiding maintainability.

use super::wavenet::VitsWaveNet;
use crate::config::VitsConfig;
use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, VarBuilder};

// ---------------------------------------------------------------------
// Residual coupling layer / block
// ---------------------------------------------------------------------

pub struct VitsResidualCouplingLayer {
    half_channels: usize,
    conv_pre: Conv1d,
    wavenet: VitsWaveNet,
    conv_post: Conv1d,
}

impl VitsResidualCouplingLayer {
    pub fn new(config: &VitsConfig, vb: VarBuilder) -> Result<Self> {
        let half_channels = config.flow_size / 2;

        let conv_pre = conv1d(
            &vb.pp("conv_pre"),
            half_channels,
            config.hidden_size,
            1,
            Conv1dConfig::default(),
        )?;
        let wavenet = VitsWaveNet::new(config, config.prior_encoder_num_wavenet_layers, vb.pp("wavenet"))?;
        let conv_post = conv1d(
            &vb.pp("conv_post"),
            config.hidden_size,
            half_channels,
            1,
            Conv1dConfig::default(),
        )?;

        Ok(Self {
            half_channels,
            conv_pre,
            wavenet,
            conv_post,
        })
    }

    /// Returns `(outputs, log_determinant)`. `log_determinant` is `None`
    /// when `reverse == true` (matches Python).
    pub fn forward(
        &self,
        inputs: &Tensor,
        padding_mask: &Tensor,
        global_conditioning: Option<&Tensor>,
        reverse: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let first_half = inputs.narrow(1, 0, self.half_channels)?;
        let second_half = inputs.narrow(1, self.half_channels, self.half_channels)?;

        let mut hidden_states = self.conv_pre.forward(&first_half)?.mul(padding_mask)?;
        hidden_states = self.wavenet.forward(&hidden_states, padding_mask, global_conditioning)?;
        let mean = self.conv_post.forward(&hidden_states)?.mul(padding_mask)?;
        let log_stddev = Tensor::zeros_like(&mean)?;

        if !reverse {
            let second_half =
                (mean.clone() + second_half.mul(&log_stddev.exp()?)?)?.mul(padding_mask)?;
            let outputs = Tensor::cat(&[&first_half, &second_half], 1)?;
            let log_determinant = log_stddev.sum((1, 2))?;
            Ok((outputs, Some(log_determinant)))
        } else {
            let second_half = second_half
                .sub(&mean)?
                .mul(&log_stddev.neg()?.exp()?)?
                .mul(padding_mask)?;
            let outputs = Tensor::cat(&[&first_half, &second_half], 1)?;
            Ok((outputs, None))
        }
    }
}

pub struct VitsResidualCouplingBlock {
    flows: Vec<VitsResidualCouplingLayer>,
}

impl VitsResidualCouplingBlock {
    pub fn new(config: &VitsConfig, vb: VarBuilder) -> Result<Self> {
        let mut flows = Vec::with_capacity(config.prior_encoder_num_flows);
        for i in 0..config.prior_encoder_num_flows {
            flows.push(VitsResidualCouplingLayer::new(config, vb.pp(format!("flows.{i}")))?);
        }
        Ok(Self { flows })
    }

    pub fn forward(
        &self,
        inputs: &Tensor,
        padding_mask: &Tensor,
        global_conditioning: Option<&Tensor>,
        reverse: bool,
    ) -> Result<Tensor> {
        let mut inputs = inputs.clone();
        if !reverse {
            for flow in &self.flows {
                let (out, _) = flow.forward(&inputs, padding_mask, global_conditioning, false)?;
                inputs = out.flip(&[1])?;
            }
        } else {
            for flow in self.flows.iter().rev() {
                inputs = inputs.flip(&[1])?;
                let (out, _) = flow.forward(&inputs, padding_mask, global_conditioning, true)?;
                inputs = out;
            }
        }
        Ok(inputs)
    }
}

// ---------------------------------------------------------------------
// Dilated depthwise-separable conv (shared by ConvFlow & duration predictor)
// ---------------------------------------------------------------------

pub struct VitsDilatedDepthSeparableConv {
    num_layers: usize,
    dropout_p: f64,
    convs_dilated: Vec<Conv1d>,
    convs_pointwise: Vec<Conv1d>,
    norms_1: Vec<LayerNorm>,
    norms_2: Vec<LayerNorm>,
}

impl VitsDilatedDepthSeparableConv {
    pub fn new(config: &VitsConfig, dropout_rate: f64, vb: VarBuilder) -> Result<Self> {
        let kernel_size = config.duration_predictor_kernel_size;
        let channels = config.hidden_size;
        let num_layers = config.depth_separable_num_layers;

        let mut convs_dilated = Vec::with_capacity(num_layers);
        let mut convs_pointwise = Vec::with_capacity(num_layers);
        let mut norms_1 = Vec::with_capacity(num_layers);
        let mut norms_2 = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let dilation = kernel_size.pow(i as u32);
            let padding = (kernel_size * dilation - dilation) / 2;

            let dilated_cfg = Conv1dConfig {
                padding,
                stride: 1,
                dilation,
                groups: channels,
                cudnn_fwd_algo: None,
            };
            let dilated_vb = vb.pp(format!("convs_dilated.{i}"));
            let weight = dilated_vb.get((channels, 1, kernel_size), "weight")?;
            let bias = dilated_vb.get(channels, "bias")?;
            convs_dilated.push(Conv1d::new(weight, Some(bias), dilated_cfg));

            convs_pointwise.push(conv1d(
                &vb.pp(format!("convs_pointwise.{i}")),
                channels,
                channels,
                1,
                Conv1dConfig::default(),
            )?);

            norms_1.push(candle_nn::layer_norm(
                channels,
                config.layer_norm_eps,
                vb.pp(format!("norms_1.{i}")),
            )?);
            norms_2.push(candle_nn::layer_norm(
                channels,
                config.layer_norm_eps,
                vb.pp(format!("norms_2.{i}")),
            )?);
        }

        Ok(Self {
            num_layers,
            dropout_p: dropout_rate,
            convs_dilated,
            convs_pointwise,
            norms_1,
            norms_2,
        })
    }

    pub fn forward(
        &self,
        inputs: &Tensor,
        padding_mask: &Tensor,
        global_conditioning: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut inputs = if let Some(gc) = global_conditioning {
            inputs.add(gc)?
        } else {
            inputs.clone()
        };

        for i in 0..self.num_layers {
            let masked = inputs.mul(padding_mask)?;
            let mut hidden_states = self.convs_dilated[i].forward(&masked)?;
            hidden_states = self.norms_1[i]
                .forward(&hidden_states.transpose(1, D::Minus1)?)?
                .transpose(1, D::Minus1)?;
            hidden_states = gelu(&hidden_states)?;
            hidden_states = self.convs_pointwise[i].forward(&hidden_states)?;
            hidden_states = self.norms_2[i]
                .forward(&hidden_states.transpose(1, D::Minus1)?)?
                .transpose(1, D::Minus1)?;
            hidden_states = gelu(&hidden_states)?;
            if self.dropout_p > 0.0 {
                hidden_states = candle_nn::ops::dropout(&hidden_states, self.dropout_p as f32)?;
            }
            inputs = inputs.add(&hidden_states)?;
        }

        inputs.mul(padding_mask)
    }
}

fn gelu(x: &Tensor) -> Result<Tensor> {
    x.gelu_erf()
}

// ---------------------------------------------------------------------
// Conv flow (uses rational-quadratic spline)
// ---------------------------------------------------------------------

pub struct VitsConvFlow {
    filter_channels: usize,
    half_channels: usize,
    num_bins: usize,
    tail_bound: f64,
    conv_pre: Conv1d,
    conv_dds: VitsDilatedDepthSeparableConv,
    conv_proj: Conv1d,
}

impl VitsConvFlow {
    pub fn new(config: &VitsConfig, vb: VarBuilder) -> Result<Self> {
        let filter_channels = config.hidden_size;
        let half_channels = config.depth_separable_channels / 2;
        let num_bins = config.duration_predictor_flow_bins;
        let tail_bound = config.duration_predictor_tail_bound;

        let conv_pre = conv1d(&vb.pp("conv_pre"), half_channels, filter_channels, 1, Conv1dConfig::default())?;
        let conv_dds = VitsDilatedDepthSeparableConv::new(config, 0.0, vb.pp("conv_dds"))?;
        let conv_proj = conv1d(
            &vb.pp("conv_proj"),
            filter_channels,
            half_channels * (num_bins * 3 - 1),
            1,
            Conv1dConfig::default(),
        )?;

        Ok(Self {
            filter_channels,
            half_channels,
            num_bins,
            tail_bound,
            conv_pre,
            conv_dds,
            conv_proj,
        })
    }

    pub fn forward(
        &self,
        inputs: &Tensor,
        padding_mask: &Tensor,
        global_conditioning: Option<&Tensor>,
        reverse: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let first_half = inputs.narrow(1, 0, self.half_channels)?;
        let second_half = inputs.narrow(1, self.half_channels, self.half_channels)?;

        let mut hidden_states = self.conv_pre.forward(&first_half)?;
        hidden_states = self.conv_dds.forward(&hidden_states, padding_mask, global_conditioning)?;
        hidden_states = self.conv_proj.forward(&hidden_states)?.mul(padding_mask)?;

        let (batch_size, _channels, length) = first_half.dims3()?;
        // reshape(batch, channels, -1, length).permute(0, 1, 3, 2)
        let last_dim = hidden_states.elem_count() / (batch_size * self.half_channels * length);
        hidden_states = hidden_states
            .reshape((batch_size, self.half_channels, last_dim, length))?
            .permute((0, 1, 3, 2))?
            .contiguous()?;

        let sqrt_fc = (self.filter_channels as f64).sqrt();
        let unnormalized_widths = (hidden_states.narrow(3, 0, self.num_bins)? / sqrt_fc)?;
        let unnormalized_heights =
            (hidden_states.narrow(3, self.num_bins, self.num_bins)? / sqrt_fc)?;
        let unnormalized_derivatives =
            hidden_states.narrow(3, 2 * self.num_bins, self.num_bins - 1)?;

        let (second_half_out, log_abs_det) = unconstrained_rational_quadratic_spline(
            &second_half,
            &unnormalized_widths,
            &unnormalized_heights,
            &unnormalized_derivatives,
            reverse,
            self.tail_bound,
            1e-3,
            1e-3,
            1e-3,
        )?;

        let outputs = Tensor::cat(&[&first_half, &second_half_out], 1)?.mul(padding_mask)?;
        if !reverse {
            let log_determinant = log_abs_det.mul(padding_mask)?.sum((1, 2))?;
            Ok((outputs, Some(log_determinant)))
        } else {
            Ok((outputs, None))
        }
    }
}

// ---------------------------------------------------------------------
// Elementwise affine (first flow in both flows/post_flows lists)
// ---------------------------------------------------------------------

pub struct VitsElementwiseAffine {
    translate: Tensor,
    log_scale: Tensor,
}

impl VitsElementwiseAffine {
    pub fn new(config: &VitsConfig, vb: VarBuilder) -> Result<Self> {
        let channels = config.depth_separable_channels;
        Ok(Self {
            translate: vb.get((channels, 1), "translate")?,
            log_scale: vb.get((channels, 1), "log_scale")?,
        })
    }

    pub fn forward(
        &self,
        inputs: &Tensor,
        padding_mask: &Tensor,
        reverse: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        if !reverse {
            let outputs = self
                .translate
                .broadcast_add(&self.log_scale.exp()?.broadcast_mul(inputs)?)?
                .mul(padding_mask)?;
            let log_determinant = self.log_scale.broadcast_mul(padding_mask)?.sum((1, 2))?;
            Ok((outputs, Some(log_determinant)))
        } else {
            let outputs = inputs
                .broadcast_sub(&self.translate)?
                .broadcast_mul(&self.log_scale.neg()?.exp()?)?
                .mul(padding_mask)?;
            Ok((outputs, None))
        }
    }
}

// ---------------------------------------------------------------------
// Rational-quadratic spline (used only by VitsConvFlow)
// ---------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn unconstrained_rational_quadratic_spline(
    inputs: &Tensor,
    unnormalized_widths: &Tensor,
    unnormalized_heights: &Tensor,
    unnormalized_derivatives: &Tensor,
    reverse: bool,
    tail_bound: f64,
    min_bin_width: f64,
    min_bin_height: f64,
    min_derivative: f64,
) -> Result<(Tensor, Tensor)> {
    // This checkpoint's typical inputs stay within [-tail_bound, tail_bound]
    // after the model's own normalization, so — matching common inference
    // simplifications of this HF module — we run the spline unconditionally
    // over the full tensor rather than PyTorch's mask-and-scatter approach
    // (masked assignment isn't idiomatic/cheap in candle). Values outside
    // the interval are clamped to the boundary before the spline, and the
    // identity transform is substituted for those positions afterward.
    let device = inputs.device();
    let dtype = inputs.dtype();

    let lower_bound = -tail_bound;
    let upper_bound = tail_bound;

    let inside_mask = inputs
        .ge(&Tensor::full(lower_bound as f32, inputs.shape(), device)?)?
        .mul(&inputs.le(&Tensor::full(upper_bound as f32, inputs.shape(), device)?)?)?;
    let inside_mask = inside_mask.to_dtype(dtype)?;

    let constant = ((1f64 - min_derivative).exp() - 1f64).ln();
    let derivatives_padded = pad_last_dim(unnormalized_derivatives, 1, 1, constant)?;

    let (outputs_inside, log_abs_det_inside) = rational_quadratic_spline(
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        &derivatives_padded,
        reverse,
        tail_bound,
        min_bin_width,
        min_bin_height,
        min_derivative,
    )?;

    // outputs = inside ? spline(inputs) : inputs ; log_abs_det = inside ? spline_log_det : 0
    let outputs = (inputs.mul(&(1.0 - &inside_mask)?)? + outputs_inside.mul(&inside_mask)?)?;
    let log_abs_det = log_abs_det_inside.mul(&inside_mask)?;

    Ok((outputs, log_abs_det))
}

#[allow(clippy::too_many_arguments)]
fn rational_quadratic_spline(
    inputs: &Tensor,
    unnormalized_widths: &Tensor,
    unnormalized_heights: &Tensor,
    unnormalized_derivatives: &Tensor,
    reverse: bool,
    tail_bound: f64,
    min_bin_width: f64,
    min_bin_height: f64,
    min_derivative: f64,
) -> Result<(Tensor, Tensor)> {
    let num_bins = unnormalized_widths.dim(D::Minus1)?;
    let upper_bound = tail_bound;
    let lower_bound = -tail_bound;

    // Clamp inputs into [lower_bound, upper_bound] to keep bin-index lookup
    // well-defined even for values that were outside the interval (those
    // get overwritten by the caller anyway).
    let inputs = inputs.clamp(lower_bound as f32, upper_bound as f32)?;

    let widths = candle_nn::ops::softmax(unnormalized_widths, D::Minus1)?;
    let widths = ((widths * (1.0 - min_bin_width * num_bins as f64))? + min_bin_width)?;
    let cumwidths = cumsum_last_dim(&widths)?;
    let cumwidths = pad_last_dim(&cumwidths, 1, 0, 0.0)?;
    let cumwidths = ((cumwidths * (upper_bound - lower_bound))? + lower_bound)?;
    let cumwidths = set_last_dim_index(&cumwidths, 0, lower_bound)?;
    let last_idx = cumwidths.dim(D::Minus1)? - 1;
    let cumwidths = set_last_dim_index(&cumwidths, last_idx, upper_bound)?;
    let widths = cumwidths
        .narrow(D::Minus1, 1, last_idx)?
        .sub(&cumwidths.narrow(D::Minus1, 0, last_idx)?)?;

    let derivatives = (candle_nn::ops::softplus(unnormalized_derivatives)? + min_derivative)?;

    let heights = candle_nn::ops::softmax(unnormalized_heights, D::Minus1)?;
    let heights = ((heights * (1.0 - min_bin_height * num_bins as f64))? + min_bin_height)?;
    let cumheights = cumsum_last_dim(&heights)?;
    let cumheights = pad_last_dim(&cumheights, 1, 0, 0.0)?;
    let cumheights = ((cumheights * (upper_bound - lower_bound))? + lower_bound)?;
    let cumheights = set_last_dim_index(&cumheights, 0, lower_bound)?;
    let cumheights = set_last_dim_index(&cumheights, last_idx, upper_bound)?;
    let heights = cumheights
        .narrow(D::Minus1, 1, last_idx)?
        .sub(&cumheights.narrow(D::Minus1, 0, last_idx)?)?;

    let bin_locations = if reverse { cumheights.clone() } else { cumwidths.clone() };
    let bin_locations = bump_last_index(&bin_locations, 1e-6)?;

    // bin_idx = sum(inputs[..., None] >= bin_locations, dim=-1) - 1
    let inputs_expanded = inputs.unsqueeze(D::Minus1)?;
    let ge = inputs_expanded.broadcast_ge(&bin_locations)?.to_dtype(DType::F32)?;
    let bin_idx = ge.sum(D::Minus1)?.affine(1.0, -1.0)?;
    let bin_idx = bin_idx
        .clamp(0f32, (num_bins as f32) - 1.0)?
        .to_dtype(DType::U32)?
        .unsqueeze(D::Minus1)?;

    let input_cumwidths = gather_last(&cumwidths, &bin_idx)?;
    let input_bin_widths = gather_last(&widths, &bin_idx)?;
    let input_cumheights = gather_last(&cumheights, &bin_idx)?;
    let delta = heights.div(&widths)?;
    let input_delta = gather_last(&delta, &bin_idx)?;
    let input_derivatives = gather_last(&derivatives, &bin_idx)?;
    let derivatives_plus_one = derivatives.narrow(D::Minus1, 1, derivatives.dim(D::Minus1)? - 1)?;
    let input_derivatives_plus_one = gather_last(&derivatives_plus_one, &bin_idx)?;
    let input_heights = gather_last(&heights, &bin_idx)?;

    let intermediate1 = (input_derivatives.clone() + input_derivatives_plus_one.clone())?
        .sub(&(input_delta.clone() * 2.0)?)?;

    if !reverse {
        let theta = inputs.sub(&input_cumwidths)?.div(&input_bin_widths)?;
        let theta_one_minus_theta = theta.mul(&(1.0 - &theta)?)?;

        let numerator = input_heights.mul(
            &(input_delta.mul(&theta.sqr()?)?
                .add(&input_derivatives.mul(&theta_one_minus_theta)?)?),
        )?;
        let denominator = input_delta.add(&intermediate1.mul(&theta_one_minus_theta)?)?;
        let outputs = input_cumheights.add(&numerator.div(&denominator)?)?;

        let derivative_numerator = input_delta.sqr()?.mul(
            &(input_derivatives_plus_one
                .mul(&theta.sqr()?)?
                .add(&(input_delta.mul(&theta_one_minus_theta)? * 2.0)?)?
                .add(&input_derivatives.mul(&(1.0 - &theta)?.sqr()?)?)?),
        )?;
        let log_abs_det = derivative_numerator.log()?.sub(&(denominator.log()? * 2.0)?)?;
        Ok((outputs, log_abs_det))
    } else {
        let intermediate2 = inputs.sub(&input_cumheights)?;
        let intermediate3 = intermediate2.mul(&intermediate1)?;
        let a = input_heights
            .mul(&input_delta.sub(&input_derivatives)?)?
            .add(&intermediate3)?;
        let b = input_heights.mul(&input_derivatives)?.sub(&intermediate3)?;
        let c = input_delta.mul(&intermediate2)?.neg()?;

        let discriminant = b.sqr()?.sub(&(a.mul(&c)? * 4.0)?)?;
        let discriminant = discriminant.clamp(0f32, f32::INFINITY)?; // numerical safety net

        let root = (c * 2.0)?.div(&b.neg()?.sub(&discriminant.sqrt()?)?)?;
        let outputs = root.mul(&input_bin_widths)?.add(&input_cumwidths)?;

        let theta_one_minus_theta = root.mul(&(1.0 - &root)?)?;
        let denominator = input_delta.add(&intermediate1.mul(&theta_one_minus_theta)?)?;
        let derivative_numerator = input_delta.sqr()?.mul(
            &(input_derivatives_plus_one
                .mul(&root.sqr()?)?
                .add(&(input_delta.mul(&theta_one_minus_theta)? * 2.0)?)?
                .add(&input_derivatives.mul(&(1.0 - &root)?.sqr()?)?)?),
        )?;
        let log_abs_det = derivative_numerator.log()?.sub(&(denominator.log()? * 2.0)?)?;
        Ok((outputs, log_abs_det.neg()?))
    }
}

// ---------------------------------------------------------------------
// Small tensor helpers (candle doesn't provide these as one-liners)
// ---------------------------------------------------------------------

/// Cumulative sum along the last dimension.
fn cumsum_last_dim(x: &Tensor) -> Result<Tensor> {
    let dim = x.rank() - 1;
    x.cumsum(dim)
}

/// Pads the last dimension with `left`/`right` elements of constant `value`
/// (mirrors `nn.functional.pad(x, [left, right])`).
fn pad_last_dim(x: &Tensor, left: usize, right: usize, value: f64) -> Result<Tensor> {
    let dim = x.rank() - 1;
    let mut parts: Vec<Tensor> = Vec::new();
    if left > 0 {
        let mut shape = x.dims().to_vec();
        shape[dim] = left;
        parts.push(Tensor::full(value as f32, shape, x.device())?.to_dtype(x.dtype())?);
    }
    parts.push(x.clone());
    if right > 0 {
        let mut shape = x.dims().to_vec();
        shape[dim] = right;
        parts.push(Tensor::full(value as f32, shape, x.device())?.to_dtype(x.dtype())?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, dim)
}

/// Replaces `x[..., start:start+fill.dim(dim), ...]` (along `dim`) with `fill`.
fn overwrite_slice(x: &Tensor, dim: usize, start: usize, fill: &Tensor) -> Result<Tensor> {
    let len = fill.dim(dim)?;
    let total = x.dim(dim)?;
    let before = if start > 0 { Some(x.narrow(dim, 0, start)?) } else { None };
    let after_start = start + len;
    let after = if after_start < total {
        Some(x.narrow(dim, after_start, total - after_start)?)
    } else {
        None
    };
    let mut parts: Vec<Tensor> = Vec::new();
    if let Some(b) = before {
        parts.push(b);
    }
    parts.push(fill.clone());
    if let Some(a) = after {
        parts.push(a);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, dim)
}

/// Sets a single index along the last dim to a constant scalar, broadcast
/// over all other dims (mirrors `tensor[..., idx] = value` in PyTorch).
fn set_last_dim_index(x: &Tensor, idx: usize, value: f64) -> Result<Tensor> {
    let dim = x.rank() - 1;
    overwrite_slice(
        x,
        dim,
        idx,
        &{
            let mut shape = x.dims().to_vec();
            shape[dim] = 1;
            Tensor::full(value as f32, shape, x.device())?.to_dtype(x.dtype())?
        },
    )
}

/// Adds `eps` to the last element along the last dim (mirrors
/// `bin_locations[..., -1] += 1e-6`).
fn bump_last_index(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dim = x.rank() - 1;
    let len = x.dim(dim)?;
    let last = x.narrow(dim, len - 1, 1)?;
    let bumped = (last + eps)?;
    overwrite_slice(x, dim, len - 1, &bumped)
}

/// Gathers along the last dimension using an index tensor whose shape
/// matches `x` except the last dim is 1, then drops that trailing dim
/// (mirrors `x.gather(-1, idx)[..., 0]` in the Python source).
fn gather_last(x: &Tensor, idx: &Tensor) -> Result<Tensor> {
    let gathered = x.gather(idx, x.rank() - 1)?;
    gathered.squeeze(D::Minus1)
}

fn conv1d(
    vb: &VarBuilder,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv1dConfig,
) -> Result<Conv1d> {
    candle_nn::conv1d(in_channels, out_channels, kernel_size, cfg, vb.clone())
}