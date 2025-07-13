use anyhow::Result;
use argh::FromArgs;
use darknet::{BBox, Image, Network};
use std::{
    fs::{self},
    path::PathBuf,
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
    #[argh(option, default = "0.9")]
    objectness_threshold: f32,
    /// the class probability threshold.
    #[argh(option, default = "0.9")]
    class_prob_threshold: f32,

    /// the URL to fetch the input image from.
    #[argh(
        option,
        default = "String::from(\"http://nlaha-voron-cam.private:1984/api/frame.jpeg?src=picam\")"
    )]
    image_url: String,
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

    loop {
        // download input image from image_url
        let response = reqwest::blocking::get(&image_url)?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to fetch image from {}: {}",
                image_url,
                response.status()
            ));
        }

        // save to file
        let image_data = response.bytes()?;
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

        let max_det = detections.iter().max_by(|a, b| {
            a.best_class(Some(class_prob_threshold))
                .unwrap_or((0, 0.0))
                .1
                .partial_cmp(
                    &b.best_class(Some(class_prob_threshold))
                        .unwrap_or((0, 0.0))
                        .1,
                )
                .unwrap_or(std::cmp::Ordering::Less)
        });
        if let Some(max_det) = max_det {
            // log
            println!(
                "Max detection probability: {:?}",
                max_det
                    .best_class(Some(class_prob_threshold))
                    .unwrap_or((0, 0.0))
                    .1
            );
        }

        detections
        .iter()
        .filter(|det| det.objectness() > objectness_threshold)
        .flat_map(|det| {
            det.best_class(Some(class_prob_threshold))
                .map(|(class_index, prob)| (det, prob, &object_labels[class_index]))
        })
        .enumerate()
        .for_each(|(index, (det, prob, label))| {
            let bbox = det.bbox();
            let BBox { x, y, w, h } = bbox;

            // if prob > 0.5 send an alert to the discord webhook
            let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");

            if prob > 0.5 {
                println!(
                    "{}: Detected {} print failure with {:.2}% confidence at x: {}, y: {}, w: {}, h: {}",
                    timestamp, label, prob * 100.0, x, y, w, h
                );
                let alert_message = format!(
                    "[{}] Alert! Detected print failure with {:.2}% confidence at x: {}, y: {}, w: {}, h: {}",
                    timestamp,
                    prob * 100.0,
                    x,
                    y,
                    w,
                    h,
                );
                let client = reqwest::blocking::Client::new();
                let _ = client.post(&discord_webhook).body(alert_message).send();
            } else {
                println!("{}: No significant print failure detected.", timestamp);
            }
        });
    }
    Ok(())
}
