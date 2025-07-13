use anyhow::Result;
use argh::FromArgs;
use darknet::{BBox, Image, Network};
use std::{
    fs::{self},
    path::PathBuf,
    thread,
    time::Duration,
};

/// Command line arguments.
#[derive(Debug, Clone, FromArgs)]
struct Args {
    /// the file including label names per class.
    #[argh(option, default = "PathBuf::from(\"./labels.txt\")")]
    label_file: PathBuf,
    /// the model config file, which usually has a .cfg extension.
    #[argh(option, default = "PathBuf::from(\"./model.cfg\")")]
    model_cfg: PathBuf,
    /// the model weights file, which usually has a .weights extension.
    #[argh(option, default = "PathBuf::from(\"./model-weights.darknet\")")]
    weights: PathBuf,
    /// the output directory.
    #[argh(option, default = "PathBuf::from(\"./output\")")]
    output_dir: PathBuf,
    /// the objectness threshold.
    #[argh(option, default = "0.5")]
    objectness_threshold: f32,
    /// the class probability threshold.
    #[argh(option, default = "0.5")]
    class_prob_threshold: f32,

    /// the URL to fetch the input image from.
    #[argh(
        option,
        default = "String::from(\"http://nlaha-voron-cam.private:1984/api/frame.jpeg?src=c920\")"
    )]
    image_url: String,
}

fn fetch_image_with_retry(
    image_url: &str,
    retry_count: &mut u32,
    max_retries: u32,
    retry_delay_seconds: u64,
    discord_webhook: &str,
    disconnect_alert_sent: &mut bool,
) -> Result<Vec<u8>> {
    loop {
        match reqwest::blocking::get(image_url) {
            Ok(response) => {
                if response.status().is_success() {
                    match response.bytes() {
                        Ok(data) => return Ok(data.to_vec()),
                        Err(e) => {
                            *retry_count += 1;
                            let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
                            println!(
                                "{}: Failed to read image data (attempt {}): {}",
                                timestamp, *retry_count, e
                            );
                        }
                    }
                } else {
                    *retry_count += 1;
                    let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
                    println!(
                        "{}: Failed to fetch image, HTTP status {} (attempt {})",
                        timestamp,
                        response.status(),
                        *retry_count
                    );
                }
            }
            Err(e) => {
                *retry_count += 1;
                let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
                println!(
                    "{}: Network error fetching image (attempt {}): {}",
                    timestamp, *retry_count, e
                );
            }
        }

        if *retry_count >= max_retries {
            // Only send the disconnect alert if we haven't sent it already
            if !*disconnect_alert_sent {
                let alert_description = format!(
                    "Failed to fetch image from {} after {} attempts. Print monitoring is offline!",
                    image_url, max_retries
                );

                // Send alert to Discord webhook
                if let Err(webhook_err) = send_discord_alert(
                    discord_webhook,
                    "CRITICAL: Print Monitoring Offline",
                    &alert_description,
                    0xFF0000, // Red color
                    "🚨"
                ) {
                    println!("Failed to send Discord alert: {}", webhook_err);
                } else {
                    println!("Sent Discord alert for connection failure");
                    *disconnect_alert_sent = true;
                }
            }

            return Err(anyhow::anyhow!(
                "Failed to fetch image after {} retries",
                max_retries
            ));
        }

        println!("Retrying in {} seconds...", retry_delay_seconds);
        thread::sleep(Duration::from_secs(retry_delay_seconds));
    }
}

/// Send a Discord alert with rich embed formatting
fn send_discord_alert(webhook_url: &str, title: &str, description: &str, color: u32, emoji: &str) -> Result<()> {
    let timestamp = chrono::Utc::now().to_rfc3339();
    
    let embed = serde_json::json!({
        "embeds": [{
            "title": format!("{} {}", emoji, title),
            "description": description,
            "color": color,
            "timestamp": timestamp,
            "footer": {
                "text": "Print Guardian"
            }
        }]
    });

    let client = reqwest::blocking::Client::new();
    let response = client
        .post(webhook_url)
        .header("Content-Type", "application/json")
        .json(&embed)
        .send()?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Failed to send Discord alert: HTTP {}",
            response.status()
        ));
    }

    Ok(())
}

fn pause_print(moonraker_api_url: &str) -> Result<()> {
    let client = reqwest::blocking::Client::new();
    let response = client
        .post(format!("{}/printer/print/pause", moonraker_api_url))
        .send()?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Failed to pause print: HTTP {}",
            response.status()
        ));
    }

    Ok(())
}

fn main() -> Result<()> {
    let Args {
        label_file,
        model_cfg,
        weights,
        output_dir,
        objectness_threshold,
        class_prob_threshold,
        image_url,
    } = argh::from_env();

    // load discord webhook URL from environment variable
    let discord_webhook =
        std::env::var("DISCORD_WEBHOOK").expect("DISCORD_WEBHOOK environment variable must be set");

    // get moonraker API URL from environment variable
    let moonraker_api_url =
        std::env::var("MOONRAKER_API_URL").expect("MOONRAKER_API_URL environment variable must be set");
    println!("Using Moonraker API URL: {}", moonraker_api_url);

    // download model weights to model-weights.darknet if it doesn't exist
    if !weights.exists() {
        let model_weights_url = "https://tsd-pub-static.s3.amazonaws.com/ml-models/model-weights-
8be06cde4e.darknet";
        let response = reqwest::blocking::get(model_weights_url)?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to download model weights from {}: {}",
                model_weights_url,
                response.status()
            ));
        }
        let model_weights_data = response.bytes()?;
        fs::write(&weights, model_weights_data)?;
    }

    // Load network & labels
    let object_labels = std::fs::read_to_string(label_file)?
        .lines()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let mut net = Network::load(model_cfg, Some(weights), false)?;

    let mut retry_count = 0;
    let mut disconnect_alert_sent = false;
    const MAX_RETRIES: u32 = 15;
    const RETRY_DELAY_SECONDS: u64 = 15;

    let mut print_failures = 0;

    loop {
        // download input image from image_url with retry logic
        let image_data = match fetch_image_with_retry(
            &image_url,
            &mut retry_count,
            MAX_RETRIES,
            RETRY_DELAY_SECONDS,
            &discord_webhook,
            &mut disconnect_alert_sent,
        ) {
            Ok(data) => {
                // If we successfully got data after being disconnected, send recovery message
                if disconnect_alert_sent {
                    if let Err(webhook_err) = send_discord_alert(
                        &discord_webhook,
                        "RECOVERY: Print Monitoring Back Online",
                        "Image fetch successful after connection issues.",
                        0x00FF00, // Green color
                        "✅"
                    ) {
                        println!("Failed to send Discord recovery alert: {}", webhook_err);
                    } else {
                        println!("Sent Discord recovery alert");
                    }
                    disconnect_alert_sent = false;
                }
                retry_count = 0; // Reset retry count on success
                data
            }
            Err(e) => {
                println!("Failed to fetch image after {} retries: {}", MAX_RETRIES, e);
                thread::sleep(Duration::from_secs(RETRY_DELAY_SECONDS));
                continue;
            }
        };

        // save to file
        let image_path = PathBuf::from("input_file.jpg");
        fs::write(&image_path, image_data)?;

        // prepare data
        let image_path = PathBuf::from("input_file.jpg");
        let image = Image::open(&image_path)?;
        let image_file_name = image_path
            .file_name()
            .expect(&format!("{} is not a valid file", image_path.display()));
        let curr_output_dir = output_dir.join(image_file_name);
        fs::create_dir_all(&curr_output_dir)?;

        // Run object detection
        let detections = net.predict(&image, 0.25, 0.5, 0.45, true);

        // print detection probabilities with timestamp
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        println!(
            "{}: Detection probability: {}",
            timestamp,
            detections
                .iter()
                // get first class probabilities
                .map(|det| det.probabilities().get(0).map_or(0.0, |p| *p))
                .collect::<Vec<_>>()
                // get max probabilities
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or("No detections".to_string(), |&p| format!(
                    "{:.2}%",
                    p * 100.0
                ))
        );

        detections
        .iter()
        .filter(|det| det.objectness() > objectness_threshold)
        .flat_map(|det| {
            det.best_class(Some(class_prob_threshold))
                .map(|(class_index, prob)| (det, prob, &object_labels[class_index]))
        })
        .enumerate()
        .for_each(|(_index, (det, prob, label))| {
            let bbox = det.bbox();
            let BBox { x, y, w, h } = bbox;

            // if prob > 0.5 send an alert to the discord webhook

            if prob > 0.5 {
                println!(
                    "{}: Detected {} print failure with {:.2}% confidence at x: {}, y: {}, w: {}, h: {}",
                    timestamp, label, prob * 100.0, x, y, w, h
                );
                
                let alert_description = format!(
                    "Detected **{}** print failure with **{:.2}%** confidence\n\n**Location:**\n• X: {:.1}\n• Y: {:.1}\n• Width: {:.1}\n• Height: {:.1}",
                    label, prob * 100.0, x, y, w, h
                );
                
                print_failures += 1;

                // Pause the print if we detect more than 3 print failures
                if print_failures > 3 {
                    if let Err(pause_err) = pause_print(&moonraker_api_url) {
                        println!("Failed to pause print: {}", pause_err);
                    } else {
                        send_discord_alert(
                            &discord_webhook,
                            "Print Paused Due to Multiple Failures",
                            &format!(
                                "Print has been paused after detecting {} print failures. Please check the printer.",
                                print_failures
                            ),
                            0xFF0000, // Red color
                            "🚨"
                        ).unwrap_or_else(|e| println!("Failed to send pause alert: {}", e));
                        println!("Print paused due to multiple failures. Alert sent to Discord.");

                        print_failures = 0; // Reset after pausing
                    }
                }

                if let Err(webhook_err) = send_discord_alert(
                    &discord_webhook,
                    "Print Failure Detected",
                    &alert_description,
                    0xFFA500, // Orange color
                    "⚠️"
                ) {
                    println!("Failed to send Discord print failure alert: {}", webhook_err);
                } else {
                    println!("Sent Discord print failure alert");
                }
            } else {
                println!("{}: No significant print failure detected.", timestamp);
            }
        });
    }
    Ok(())
}
