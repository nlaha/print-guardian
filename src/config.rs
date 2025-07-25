use std::path::PathBuf;

/// Configuration for the Print Guardian application loaded from environment variables.
///
/// This struct defines all the configurable parameters for the print monitoring system,
/// including model paths, thresholds, and input sources. All values are loaded from
/// environment variables to support containerized deployments.
#[derive(Debug, Clone)]
pub struct Config {
    /// The file including label names per class.
    ///
    /// This file should contain one label per line, corresponding to the classes
    /// that the neural network can detect (e.g., "spaghetti", "layer_shift", etc.).
    /// Environment variable: `LABEL_FILE`
    pub label_file: PathBuf,

    /// The model config file, which usually has a .cfg extension.
    ///
    /// This is the YOLO/Darknet configuration file that defines the neural network
    /// architecture used for object detection.
    /// Environment variable: `MODEL_CFG`
    pub model_cfg: PathBuf,

    /// The model weights file, which usually has a .weights extension.
    ///
    /// Contains the trained weights for the neural network. If this file doesn't exist,
    /// it will be automatically downloaded from the configured URL.
    /// Environment variable: `WEIGHTS_FILE`
    pub weights: PathBuf,

    /// The output directory for saving processed images and detection results.
    /// Environment variable: `OUTPUT_DIR`
    pub output_dir: PathBuf,

    /// The objectness threshold for object detection.
    ///
    /// Objects with objectness scores below this threshold will be filtered out.
    /// Higher values result in fewer, but more confident detections.
    /// Environment variable: `OBJECTNESS_THRESHOLD`
    pub objectness_threshold: f32,

    /// The class probability threshold for classification.
    ///
    /// Detections with class probabilities below this threshold will be ignored.
    /// This helps reduce false positives by only considering high-confidence classifications.
    /// Environment variable: `CLASS_PROB_THRESHOLD`
    pub class_prob_threshold: f32,

    /// The URLs to fetch input images from.
    ///
    /// This can be a single URL or multiple comma-separated URLs pointing to
    /// camera feeds or image endpoints that provide real-time images of the 3D printer.
    /// When multiple URLs are provided, they will be used in round-robin fashion.
    /// Environment variable: `IMAGE_URL`
    pub image_urls: Vec<String>,

    /// Discord webhook URL for sending alerts.
    ///
    /// This should be a valid Discord webhook URL where alerts about print
    /// failures and system status will be sent.
    /// Environment variable: `DISCORD_WEBHOOK`
    pub discord_webhook: String,

    /// Moonraker API URL for printer control.
    ///
    /// This should point to the Moonraker API endpoint (typically on port 7125)
    /// that allows the application to pause prints when failures are detected.
    /// Environment variable: `MOONRAKER_API_URL`
    pub moonraker_api_url: String,

    /// Whether to flip the image vertically after fetching.
    ///
    /// This is useful for cameras that are mounted upside down.
    /// Set to "true" to enable vertical flipping.
    /// Environment variable: `FLIP_IMAGE`
    pub flip_image: bool,

    /// Optional index of the camera to display.
    /// This is used to select a specific camera
    /// when multiple cameras are configured. If not set,
    /// the first camera in the list will be used.
    pub display_camera_index: Option<usize>,
}

impl Config {
    /// Load configuration from environment variables.
    ///
    /// # Errors
    ///
    /// Returns an error if required environment variables are not set or cannot be parsed:
    /// - `LABEL_FILE`: Path to the label file (default: "./labels.txt")
    /// - `MODEL_CFG`: Path to the model config file (default: "./model.cfg")
    /// - `WEIGHTS_FILE`: Path to the weights file (default: "./model/model-weights.darknet")
    /// - `OUTPUT_DIR`: Output directory path (default: "./output")
    /// - `OBJECTNESS_THRESHOLD`: Objectness threshold (default: "0.75")
    /// - `CLASS_PROB_THRESHOLD`: Class probability threshold (default: "0.75")
    /// - `IMAGE_URL`: Camera image URL(s) (required) - single URL or comma-separated list for round-robin
    /// - `DISCORD_WEBHOOK`: Discord webhook URL (required)
    /// - `MOONRAKER_API_URL`: Moonraker API URL (required)
    /// - `FLIP_IMAGE`: Whether to flip images horizontally (default: "false")
    /// - `DISPLAY_CAMERA_INDEX`: Optional index of the camera to display (default: 0)
    ///
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let label_file = PathBuf::from(
            std::env::var("LABEL_FILE").unwrap_or_else(|_| "./labels.txt".to_string()),
        );

        let model_cfg =
            PathBuf::from(std::env::var("MODEL_CFG").unwrap_or_else(|_| "./model.cfg".to_string()));

        let weights = PathBuf::from(
            std::env::var("WEIGHTS_FILE")
                .unwrap_or_else(|_| "./model/model-weights.darknet".to_string()),
        );

        let output_dir =
            PathBuf::from(std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "./output".to_string()));

        let objectness_threshold = std::env::var("OBJECTNESS_THRESHOLD")
            .unwrap_or_else(|_| "0.75".to_string())
            .parse::<f32>()
            .map_err(|e| format!("Invalid OBJECTNESS_THRESHOLD: {}", e))?;

        let class_prob_threshold = std::env::var("CLASS_PROB_THRESHOLD")
            .unwrap_or_else(|_| "0.75".to_string())
            .parse::<f32>()
            .map_err(|e| format!("Invalid CLASS_PROB_THRESHOLD: {}", e))?;

        let image_url =
            std::env::var("IMAGE_URL").map_err(|_| "IMAGE_URL environment variable is required")?;

        // Parse image URLs - can be single URL or comma-separated list
        let image_urls: Vec<String> = image_url
            .split(',')
            .map(|url| url.trim().to_string())
            .filter(|url| !url.is_empty())
            .collect();

        if image_urls.is_empty() {
            return Err("IMAGE_URL must contain at least one valid URL".into());
        }

        let discord_webhook = std::env::var("DISCORD_WEBHOOK")
            .map_err(|_| "DISCORD_WEBHOOK environment variable is required")?;

        let moonraker_api_url = std::env::var("MOONRAKER_API_URL")
            .map_err(|_| "MOONRAKER_API_URL environment variable is required")?;

        let flip_image = std::env::var("FLIP_IMAGE")
            .unwrap_or_else(|_| "false".to_string())
            .parse::<bool>()
            .map_err(|e| format!("Invalid FLIP_IMAGE (must be 'true' or 'false'): {}", e))?;

        let display_camera_index = std::env::var("DISPLAY_CAMERA_INDEX")
            .ok()
            .unwrap_or_else(|| "0".to_string())
            .parse::<usize>()
            .ok()
            .filter(|&i| i < image_urls.len());

        Ok(Config {
            label_file,
            model_cfg,
            weights,
            output_dir,
            objectness_threshold,
            class_prob_threshold,
            image_urls,
            discord_webhook,
            moonraker_api_url,
            flip_image,
            display_camera_index,
        })
    }
}

/// Application constants used throughout the system.
pub mod constants {
    /// Maximum number of retry attempts when fetching images fails.
    pub const MAX_RETRIES: u32 = 15;

    /// Delay between retry attempts in seconds.
    pub const RETRY_DELAY_SECONDS: u64 = 15;

    /// Threshold for print failure count before pausing the printer.
    pub const PRINT_FAILURE_THRESHOLD: u32 = 3;

    /// Probability threshold for triggering alerts (0.0 to 1.0).
    pub const ALERT_PROBABILITY_THRESHOLD: f32 = 0.5;

    /// URL for downloading model weights if they don't exist locally.
    pub const MODEL_WEIGHTS_URL: &str =
        "https://tsd-pub-static.s3.amazonaws.com/ml-models/model-weights-8be06cde4e.darknet";
}
