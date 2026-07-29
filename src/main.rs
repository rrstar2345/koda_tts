use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use kan_tts_rs::config::VitsConfig;
use kan_tts_rs::model::VitsModel;
use kan_tts_rs::tokenizer::VitsTokenizer;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_MODEL_DIR: &str = "/home/kkvm/Projects/tts/kan_tts_rs/models/";
const DEFAULT_OUTPUT_FILE: &str = "./out/output.wav";

fn main() -> Result<()> {
    let args = CliArgs::parse()?;
    let model_dir = args.model_dir;
    let output_file = args.output_file;
    let text = args.text;

    let config = VitsConfig::from_pretrained(&model_dir)?;
    let tokenizer = VitsTokenizer::from_pretrained(&model_dir)?;
    let device = Device::Cpu;

    let model_path = model_dir.join("model.safetensors");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_path.as_path()], DType::F32, &device)
    }
    .with_context(|| format!("loading {}", model_path.display()))?;
    let model = VitsModel::new(&config, vb)?;

    let (input_ids, attention_mask) = tokenizer.call(&text)?;
    let seq_len = input_ids.len();
    let input_ids = Tensor::from_vec(
        input_ids.iter().map(|&v| v as u32).collect::<Vec<_>>(),
        (1, seq_len),
        &device,
    )?;
    let attention_mask = Tensor::from_vec(attention_mask, (1, seq_len), &device)?;

    let output = model.forward(&input_ids, Some(&attention_mask), None)?;
    let waveform = output.waveform.squeeze(0)?;
    let samples: Vec<f32> = waveform.to_vec1()?;
    write_wav(&output_file, &samples, config.sampling_rate)?;

    println!(
        "Saved {} ({} samples @ {} Hz)",
        output_file.display(),
        samples.len(),
        config.sampling_rate
    );
    Ok(())
}

struct CliArgs {
    model_dir: PathBuf,
    text: String,
    output_file: PathBuf,
}

impl CliArgs {
    fn parse() -> Result<Self> {
        let args = env::args().skip(1).collect::<Vec<_>>();
        if args.len() > 1 {
            anyhow::bail!(
                "usage: kan_tts [text]\n\
                 defaults: model_dir={}, file={}",
                DEFAULT_MODEL_DIR,
                DEFAULT_OUTPUT_FILE
            );
        }

        let text = args
            .get(0)
            .cloned()
            .unwrap_or_else(|| "ನನಗೆ ಕನ್ನಡ ಬರುತ್ತದೆ.".to_string());

        Ok(Self {
            model_dir: PathBuf::from(DEFAULT_MODEL_DIR),
            text,
            output_file: PathBuf::from(DEFAULT_OUTPUT_FILE),
        })
    }
}

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
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
