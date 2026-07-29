use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use chrono::Local;
use clap::Parser;
use tts_rs::config::VitsConfig;
use tts_rs::download;
use tts_rs::model::VitsModel;
use tts_rs::tokenizer::VitsTokenizer;
use tts_rs::settings::setup_app_folders;
use tts_rs::wav::write_wav;
use std::path::{PathBuf};
use std::sync::Arc;

/// A lightweight Text-to-Speech CLI application using Rust and Candle.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct CliArgs {
    /// The text you want to synthesize into speech
    #[arg(default_value = "ನನಗೆ ಕನ್ನಡ ಬರುತ್ತದೆ.")]
    text: String,

    /// Language option ("kan", "tam", "eng")
    #[arg(short, long, default_value = "kan")]
    language: String,

    /// Path to save the generated output WAV file (auto-generated if not provided)
    #[arg(short, long)]
    output_file: Option<PathBuf>,
}

fn main() -> Result<()> {
    // Parse command line arguments via clap
    let args = CliArgs::parse();

    // Process language input: take first character, lowercase, and validate
    let raw_lang = args.language.trim();
    let normalized_lang = if let Some(first_char) = raw_lang.chars().next() {
        first_char.to_lowercase().to_string()
    } else {
        String::new()
    };

    let tts_lang = match normalized_lang.as_str() {
        "k" => "kan",
        "t" => "tam",
        "e" => "eng",
        _ => {
            eprintln!(
                "Error: The app does not support the language '{}' yet. Supported options are Kannada ('kan'), Tamil ('tam'), and English ('eng').",
                args.language
            );
            std::process::exit(1);
        }
    };

    // Set up application folders and automatically fetch the model directory path based on language
    let paths = Arc::new(setup_app_folders()?);
    let model_dir = download::download_models(&paths, tts_lang)?;

    let text = args.text;

    // Determine output file path: generate a unique name if not provided
    let output_file = match args.output_file {
        Some(path) => path,
        None => {
            let timestamp = Local::now().format("%Y%m%d-%H%M%S").to_string();
            let file_name = format!("{}-{}.wav", tts_lang, timestamp);
            PathBuf::from("./out").join(file_name)
        }
    };

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