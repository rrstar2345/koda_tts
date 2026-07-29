use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }

    let mut pcm = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let clipped = sample.clamp(-1.0, 1.0);
        let quantized = (clipped * i16::MAX as f32).round() as i16;
        pcm.extend_from_slice(&quantized.to_le_bytes());
    }

    let data_len = pcm.len() as u32;
    let riff_len = 36u32 + data_len;
    let mut wav = Vec::with_capacity((riff_len + 8) as usize);

    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&riff_len.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * 2;
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(&pcm);

    fs::write(path, wav).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}