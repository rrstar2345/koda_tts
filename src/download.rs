use crate::settings::AppPaths;
use std::fs::{self, File};
use std::io::{self, Read, Write, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const TTS_MODEL_FILE: &str = "model.safetensors";
pub const TTS_MODEL_CONFIG_FILE: &str = "config.json";
pub const TTS_TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";
pub const TTS_MODEL_VOCAB_FILE: &str = "vocab.json";

/// Download model and config files.
pub fn download_models(app_dir: &AppPaths, tts_lang: &str) -> io::Result<PathBuf> {
    let agent = ureq::Agent::new_with_defaults();
    
    let tts_dir = app_dir.tts.join(format!("mms-tts-{}", tts_lang));
    fs::create_dir_all(&tts_dir)?;
    
    let tts_base = format!("https://huggingface.co/facebook/mms-tts-{}/resolve/main", tts_lang);

    for name in [TTS_MODEL_FILE, TTS_MODEL_CONFIG_FILE, TTS_TOKENIZER_CONFIG_FILE, TTS_MODEL_VOCAB_FILE] {
        let dest = tts_dir.join(name);
        if !dest.exists() {
            let url = format!("{}/{}", tts_base, name);
            download(&agent, &url, &dest)?;
        }
    }

    Ok(tts_dir)
}

fn download(agent: &ureq::Agent, url: &str, dest: &Path) -> io::Result<()> {
    eprintln!("Downloading: {}", url);
    let resp = agent
        .get(url)
        .header("User-Agent", "lite-tts/0.1")
        .call()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{e}")))?;

    let total_len: Option<u64> = resp
        .headers()
        .get("content-length")
        .and_then(|hv| hv.to_str().ok())
        .and_then(|s| s.parse().ok());

    let tmp = dest.with_extension("part");
    let mut file = File::create(&tmp)?;
    let started = Instant::now();

    let mut body = resp.into_body();
    let mut reader = BufReader::new(body.as_reader());
    let mut chunk = vec![0u8; 256 * 1024];
    let mut copied: u64 = 0;
    let mut last_logged = 0u64;

    loop {
        let n = reader.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        file.write_all(&chunk[..n])?;
        copied += n as u64;
        if copied - last_logged >= 8 * 1024 * 1024 {
            display_progress(copied, total_len, started);
            last_logged = copied;
        }
    }

    display_progress(copied, total_len, started);
    eprintln!();
    file.sync_all()?;
    drop(file);
    fs::rename(&tmp, dest)?;
    Ok(())
}

fn display_progress(copied: u64, total_len: Option<u64>, started: Instant) {
    let mb = copied as f64 / (1024.0 * 1024.0);
    let secs = started.elapsed().as_secs_f64().max(1e-3);
    let mbps = mb / secs;

    match total_len {
        Some(total) => {
            let total_mb = total as f64 / (1024.0 * 1024.0);
            let pct = (copied as f64 / total as f64) * 100.0;
            let bar_width = 30;
            let filled = (pct as usize * bar_width / 100).min(bar_width);
            let bar = "█".repeat(filled) + &"░".repeat(bar_width - filled);
            eprint!(
                "\r  [{}] {:.0}% | {:.1} MB / {:.1} MB | {:.1} MB/s",
                bar, pct, mb, total_mb, mbps
            );
        }
        None => {
            eprint!("\r  ⤓ {:.1} MB | {:.1} MB/s", mb, mbps);
        }
    }
    let _ = io::stderr().flush();
}