use anyhow::Result;
use darknet::{BBox, Image, Network};
use log::{debug, info};
use serde_json::de;
use std::{fs, path::PathBuf};

/// Print failure detection service using YOLO/Darknet neural networks.
///
/// This service handles loading and running object detection models to identify
/// print failures
pub struct FailureDetector {
    network: Network,
    labels: Vec<String>,
    objectness_threshold: f32,
    class_prob_threshold: f32,
}

impl FailureDetector {
    /// Create a new FailureDetector with the specified model and configuration.
    ///
    /// # Arguments
    ///
    /// * `model_cfg` - Path to the YOLO/Darknet configuration file
    /// * `weights_path` - Path to the trained model weights file
    /// * `labels_path` - Path to the file containing class labels
    /// * `objectness_threshold` - Minimum objectness score for detections
    /// * `class_prob_threshold` - Minimum class probability for valid detections
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model files cannot be loaded
    /// - Labels file cannot be read
    /// - Model configuration is invalid
    ///
    /// # Examples
    ///
    /// ```rust
    /// let detector = FailureDetector::new(
    ///     PathBuf::from("model.cfg"),
    ///     PathBuf::from("weights.darknet"),
    ///     PathBuf::from("labels.txt"),
    ///     0.5,
    ///     0.5
    /// )?;
    /// ```
    pub fn new(
        model_cfg: PathBuf,
        weights_path: PathBuf,
        labels_path: PathBuf,
        objectness_threshold: f32,
        class_prob_threshold: f32,
    ) -> Result<Self> {
        // Load class labels
        let labels = fs::read_to_string(labels_path)?
            .lines()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();

        // print labels file content
        info!(
            "Loaded labels: {:?}",
            labels
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );

        // Load the neural network
        let network = Network::load(model_cfg, Some(weights_path), false)?;

        Ok(Self {
            network,
            labels,
            objectness_threshold,
            class_prob_threshold,
        })
    }

    /// Download model weights if they don't exist locally.
    ///
    /// Automatically downloads the trained model weights from a remote URL
    /// if the weights file doesn't exist on the local filesystem.
    ///
    /// # Arguments
    ///
    /// * `weights_path` - Local path where weights should be stored
    /// * `download_url` - URL to download weights from
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Download fails
    /// - File cannot be written to disk
    /// - Remote server returns error status
    pub fn ensure_weights_downloaded(weights_path: &PathBuf, download_url: &str) -> Result<()> {
        // download model weights to model-weights.darknet if it doesn't exist
        if !weights_path.exists() {
            info!(
                "Model weights not found at {}, downloading from {}",
                weights_path.display(),
                download_url
            );
            let response = reqwest::blocking::get(download_url)?;
            if !response.status().is_success() {
                return Err(anyhow::anyhow!(
                    "Failed to download model weights from {}: {}",
                    download_url,
                    response.status()
                ));
            }
            let model_weights_data = response.bytes()?;
            fs::write(weights_path, model_weights_data)?;
        } else {
            info!("Model weights already exist at {}", weights_path.display());
        }

        info!("Model weights are ready at {}", weights_path.display());

        Ok(())
    }

    /// Run failure detection on a darknet Image object directly.
    ///
    /// This method accepts a darknet Image object directly, bypassing file I/O.
    /// This is more efficient when image data is already loaded in memory.
    ///
    /// # Arguments
    ///
    /// * `image` - A darknet Image object
    ///
    /// # Returns
    ///
    /// Returns a vector of `Detection` objects containing information about
    /// each detected print failure.
    ///
    /// # Errors
    ///
    /// Returns an error if neural network inference fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let image_data = fetcher.fetch_with_retry(alert_callback)?;
    /// let darknet_image = ImageFetcher::bytes_to_darknet_image(&image_data)?;
    /// let detections = detector.detect_failures_from_image(&darknet_image)?;
    /// ```
    pub fn detect_failures_from_image(&mut self, image: &Image) -> Result<Vec<Detection>> {
        debug!(
            "Processing image with dimensions: {}x{}x{}",
            image.width(),
            image.height(),
            image.channels()
        );

        debug!("First 10 pixels: {:?}", &image.get_data()[..10]);
        let min = image
            .get_data()
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let max = image
            .get_data()
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mean = image.get_data().iter().cloned().sum::<f32>() / image.get_data().len() as f32;
        debug!("Pixel stats: min={}, max={}, mean={}", min, max, mean);

        // Run object detection with NMS parameters
        let detections = self.network.predict(image, 0.25, 0.5, 0.45, true);

        debug!("Raw detections count: {}", detections.len());
        debug!(
            "Raw detection objectness: {:?}",
            detections
                .iter()
                .map(|d| d.objectness())
                .filter(|&o| o > 0.0)
                .collect::<Vec<_>>()
        );
        debug!(
            "Raw detection classes: {:?}",
            detections
                .iter()
                .map(|d| d.best_class(None))
                .filter(|c| c.is_some() && c.unwrap().1 > 0.0)
                .collect::<Vec<_>>()
        );

        let mut results = Vec::new();

        // Process detections and filter by thresholds
        for det in detections.iter() {
            if det.objectness() > self.objectness_threshold {
                // Get the best class without threshold filtering for debugging
                if let Some((class_index, prob)) = det.best_class(None) {
                    let label = self
                        .labels
                        .get(class_index)
                        .unwrap_or(&"unknown".to_string())
                        .clone();

                    // Only add to results if it meets the threshold
                    if prob > self.class_prob_threshold {
                        let bbox = *det.bbox();

                        results.push(Detection {
                            label,
                            confidence: prob,
                            bbox,
                        });
                    }
                }
            }
        }

        Ok(results)
    }

    /// Get the list of class labels.
    pub fn get_labels(&self) -> &[String] {
        &self.labels
    }

    /// Get the current objectness threshold.
    pub fn get_objectness_threshold(&self) -> f32 {
        self.objectness_threshold
    }

    /// Get the current class probability threshold.
    pub fn get_class_prob_threshold(&self) -> f32 {
        self.class_prob_threshold
    }

    /// Update the objectness threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - New objectness threshold (0.0 to 1.0)
    pub fn set_objectness_threshold(&mut self, threshold: f32) {
        self.objectness_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Update the class probability threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - New class probability threshold (0.0 to 1.0)
    pub fn set_class_prob_threshold(&mut self, threshold: f32) {
        self.class_prob_threshold = threshold.clamp(0.0, 1.0);
    }
}

/// Represents a single print failure detection result.
///
/// Contains all the information about a detected print failure, including
/// the type of failure, confidence level, and location in the image.
#[derive(Debug, Clone)]
pub struct Detection {
    /// The type of print failure detected (for the included model this is only "failure").
    pub label: String,

    /// Confidence score from 0.0 to 1.0.
    pub confidence: f32,

    /// Bounding box coordinates and dimensions.
    pub bbox: BBox,
}

impl Detection {
    /// Get the bounding box center X coordinate.
    pub fn center_x(&self) -> f32 {
        self.bbox.x
    }

    /// Get the bounding box center Y coordinate.
    pub fn center_y(&self) -> f32 {
        self.bbox.y
    }

    /// Get the bounding box width.
    pub fn width(&self) -> f32 {
        self.bbox.w
    }

    /// Get the bounding box height.
    pub fn height(&self) -> f32 {
        self.bbox.h
    }

    /// Get the confidence as a percentage.
    pub fn confidence_percent(&self) -> f32 {
        self.confidence * 100.0
    }

    /// Check if this detection exceeds the specified confidence threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Confidence threshold to check against (0.0 to 1.0)
    pub fn exceeds_threshold(&self, threshold: f32) -> bool {
        self.confidence > threshold
    }
}
