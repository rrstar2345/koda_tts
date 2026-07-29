//! Small extension trait providing `Tensor::flip`, which candle-core 0.8
//! does not ship. Implemented via `index_select` with a reversed index
//! tensor, matching `torch.flip(x, dims)` semantics for the dims we need
//! (always a single dim in this codebase).

use candle_core::{DType, Result, Tensor};

pub trait FlipExt {
    /// Reverses the tensor along each dim listed in `dims`.
    fn flip(&self, dims: &[usize]) -> Result<Tensor>;
}

impl FlipExt for Tensor {
    fn flip(&self, dims: &[usize]) -> Result<Tensor> {
        let mut out = self.clone();
        for &dim in dims {
            let out_contiguous = out.contiguous()?;
            let len = out_contiguous.dim(dim)?;
            let indices: Vec<u32> = (0..len as u32).rev().collect();
            let idx = Tensor::from_vec(indices, len, out_contiguous.device())?.to_dtype(DType::U32)?;
            out = out_contiguous.index_select(&idx, dim)?;
        }
        Ok(out)
    }
}
