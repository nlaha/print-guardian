use anyhow::Result;
use log::{debug, error, info, warn};
use std::{fs, path::PathBuf, thread, time::Duration};

// Module declarations
mod alerts;
mod config;
mod detector;
mod error;
mod image_fetcher;
mod printer;

// Import our modules
use alerts::AlertService;
use config::{Config, constants};
use detector::FailureDetector;
use image_fetcher::{AlertType, ImageFetcher};
use printer::PrinterService;

/// Print Guardian - AI-powered 3D print failure detection system.
///
/// This application monitors 3D printer cameras in real-time using machine learning
/// to detect print failures such as spaghetti, layer shifts, warping, and other issues.
/// When failures are detected, it can automatically pause the printer and send alerts
/// via Discord webhooks.
///
/// # Features
///
/// * Real-time camera monitoring with retry logic
/// * AI-powered print failure detection using YOLO/Darknet
/// * Automatic printer pause on multiple failures
/// * Discord webhook notifications with rich embeds
/// * Configurable detection thresholds
/// * Robust error handling and recovery
///
/// # Environment Variables
///
/// Required:
/// * `IMAGE_URL` - Camera image URL for monitoring
/// * `DISCORD_WEBHOOK` - Discord webhook URL for alerts
/// * `MOONRAKER_API_URL` - Moonraker API endpoint for printer control
///
/// Optional (with defaults):
/// * `LABEL_FILE` - Path to label file (default: "./labels.txt")
/// * `MODEL_CFG` - Path to model config file (default: "./model.cfg")
/// * `WEIGHTS_FILE` - Path to weights file (default: "./model-weights.darknet")
/// * `OUTPUT_DIR` - Output directory (default: "./output")
/// * `OBJECTNESS_THRESHOLD` - Objectness threshold (default: "0.5")
/// * `CLASS_PROB_THRESHOLD` - Class probability threshold (default: "0.5")
///
/// # Usage
///
/// ```bash
/// export IMAGE_URL="http://camera.local/image.jpg"
/// export DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
/// export MOONRAKER_API_URL="http://printer.local:7125"
/// ./print-guardian
/// ```
fn main() -> Result<()> {
    // Initialize logger
    env_logger::init();

    // Load configuration from environment variables
    let config = Config::load().expect(
        "Failed to load configuration. Please ensure all required environment variables are set.",
    );

    info!("Print Guardian starting...");
    info!("Using Moonraker API URL: {}", config.moonraker_api_url);
    info!("Monitoring camera: {}", config.image_url);

    // Initialize services
    let alert_service = AlertService::new(config.discord_webhook.clone());
    let printer_service = PrinterService::new(config.moonraker_api_url.clone());

    // Download model weights if needed
    FailureDetector::ensure_weights_downloaded(&config.weights, constants::MODEL_WEIGHTS_URL)?;

    // Initialize failure detector
    let mut detector = FailureDetector::new(
        config.model_cfg.clone(),
        config.weights.clone(),
        config.label_file.clone(),
        config.objectness_threshold,
        config.class_prob_threshold,
    )?;

    // Initialize image fetcher
    let mut image_fetcher = ImageFetcher::new(
        config.image_url.clone(),
        constants::MAX_RETRIES,
        constants::RETRY_DELAY_SECONDS,
    );

    // Create output directory
    fs::create_dir_all(&config.output_dir)?;

    // Main monitoring loop state
    let mut print_failures = 0;

    info!("Print Guardian initialized successfully. Starting monitoring loop...");

    loop {
        // First check if the printer is online
        if let Err(e) = printer_service.check_printer_status() {
            warn!(
                "Printer is offline, retrying in {} seconds: {}",
                constants::RETRY_DELAY_SECONDS,
                e
            );
            thread::sleep(Duration::from_secs(constants::RETRY_DELAY_SECONDS));
            continue;
        }

        // Fetch image with retry logic
        let image_data = match image_fetcher.fetch_with_retry(|alert_type| match alert_type {
            AlertType::SystemOffline => alert_service.send_system_offline_alert(
                image_fetcher.get_image_url(),
                image_fetcher.get_max_retries(),
            ),
            AlertType::SystemRecovery => alert_service.send_system_recovery_alert(),
        }) {
            Ok(data) => data,
            Err(e) => {
                error!("Failed to fetch image: {}", e);
                thread::sleep(Duration::from_secs(constants::RETRY_DELAY_SECONDS));
                continue;
            }
        };

        // Save image to disk for processing
        let image_path = PathBuf::from("input_file.jpg");
        if let Err(e) = fs::write(&image_path, image_data) {
            error!("Failed to save image: {}", e);
            continue;
        }

        // Log detection probability
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        match detector.get_max_detection_probability(&image_path) {
            Ok(prob_str) => {
                debug!("{}: Detection probability: {}", timestamp, prob_str);
            }
            Err(e) => {
                error!("{}: Failed to get detection probability: {}", timestamp, e);
                continue;
            }
        }

        // Run failure detection
        let detections = match detector.detect_failures(&image_path) {
            Ok(detections) => detections,
            Err(e) => {
                error!("{}: Detection failed: {}", timestamp, e);
                continue;
            }
        };

        // Process detections
        for detection in detections {
            if detection.exceeds_threshold(constants::ALERT_PROBABILITY_THRESHOLD) {
                warn!(
                    "{}: Detected {} print failure with {:.2}% confidence at x: {:.1}, y: {:.1}, w: {:.1}, h: {:.1}",
                    timestamp,
                    detection.label,
                    detection.confidence_percent(),
                    detection.center_x(),
                    detection.center_y(),
                    detection.width(),
                    detection.height()
                );

                print_failures += 1;

                // Check if we should pause the printer
                if print_failures > constants::PRINT_FAILURE_THRESHOLD {
                    match printer_service.pause_print() {
                        Ok(()) => {
                            if let Err(e) = alert_service.send_print_pause_alert(print_failures) {
                                error!("Failed to send pause alert: {}", e);
                            } else {
                                info!(
                                    "Print paused due to multiple failures. Alert sent to Discord."
                                );
                            }
                            print_failures = 0; // Reset after pausing
                        }
                        Err(e) => {
                            error!("Failed to pause print: {}", e);
                        }
                    }
                }

                // Send failure alert
                if let Err(e) = alert_service.send_print_failure_alert(
                    &detection.label,
                    detection.confidence_percent(),
                    detection.center_x(),
                    detection.center_y(),
                    detection.width(),
                    detection.height(),
                ) {
                    error!("Failed to send Discord print failure alert: {}", e);
                } else {
                    info!("Sent Discord print failure alert");
                }
            } else {
                debug!("{}: No significant print failure detected.", timestamp);
            }
        }

        // Small delay before next iteration
        thread::sleep(Duration::from_secs(1));
    }
}
