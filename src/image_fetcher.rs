#![allow(dead_code)]
use anyhow::Result;
use log::{error, info, warn};
use std::{thread, time::Duration};

// Add imports for image processing and annotation
use darknet::Image;
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;

use crate::detector::Detection;

/// Image fetching service with retry logic and error handling.
///
/// This service handles downloading images from camera endpoints with robust
/// retry logic, timeout handling, and status tracking for reliable operation
/// in network-unstable environments. Supports multiple URLs with round-robin
/// multiplexing for load balancing or redundancy.
pub struct ImageFetcher {
    image_urls: Vec<String>,
    current_url_index: usize,
    max_retries: u32,
    retry_delay_seconds: u64,
    retry_count: u32,
    disconnect_alert_sent: bool,
}

impl ImageFetcher {
    /// Create a new ImageFetcher with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `image_urls` - Vector of URLs to fetch images from (round-robin)
    /// * `max_retries` - Maximum number of retry attempts before giving up
    /// * `retry_delay_seconds` - Delay between retry attempts
    pub fn new(image_urls: Vec<String>, max_retries: u32, retry_delay_seconds: u64) -> Self {
        Self {
            image_urls,
            current_url_index: 0,
            max_retries,
            retry_delay_seconds,
            retry_count: 0,
            disconnect_alert_sent: false,
        }
    }

    /// Fetch an image with automatic retry logic.
    ///
    /// Attempts to download an image from the configured URL. If the download
    /// fails, it will retry up to `max_retries` times with delays between attempts.
    /// Tracks connection status and can trigger alerts when disconnected or recovered.
    ///
    /// # Arguments
    ///
    /// * `alert_callback` - Callback function for sending alerts (system offline/recovery)
    ///
    /// # Returns
    ///
    /// Returns the image data as a byte vector on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - All retry attempts are exhausted
    /// - Network connectivity issues persist
    /// - The remote server returns error responses
    ///
    pub fn fetch_with_retry<F>(
        &mut self,
        mut alert_callback: F,
        url_index: Option<usize>,
    ) -> Result<Vec<u8>>
    where
        F: FnMut(AlertType) -> Result<()>,
    {
        loop {
            match self.attempt_fetch(url_index) {
                Ok(data) => {
                    // If we successfully got data after being disconnected, send recovery message
                    if self.disconnect_alert_sent {
                        if let Err(e) = alert_callback(AlertType::SystemRecovery) {
                            error!("Failed to send recovery alert: {}", e);
                        } else {
                            info!("Sent system recovery alert");
                        }
                        self.disconnect_alert_sent = false;
                    }
                    self.retry_count = 0; // Reset retry count on success
                    return Ok(data);
                }
                Err(e) => {
                    self.retry_count += 1;
                    let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
                    warn!(
                        "{}: Failed to fetch image (attempt {}): {}",
                        timestamp, self.retry_count, e
                    );

                    if self.retry_count >= self.max_retries {
                        // Only send the disconnect alert if we haven't sent it already
                        if !self.disconnect_alert_sent {
                            if let Err(alert_err) = alert_callback(AlertType::SystemOffline) {
                                error!("Failed to send disconnect alert: {}", alert_err);
                            } else {
                                info!("Sent system offline alert");
                                self.disconnect_alert_sent = true;
                            }
                        }

                        return Err(anyhow::anyhow!(
                            "Failed to fetch image after {} retries",
                            self.max_retries
                        ));
                    }

                    info!("Retrying in {} seconds...", self.retry_delay_seconds);
                    thread::sleep(Duration::from_secs(self.retry_delay_seconds));
                }
            }
        }
    }

    /// Attempt a single image fetch operation.
    ///
    /// Makes a single HTTP request to fetch an image without retry logic.
    /// This is used internally by `fetch_with_retry` but can also be used
    /// directly for testing or when custom retry logic is needed.
    ///
    /// Uses round-robin URL selection when multiple URLs are configured.
    ///
    /// # Returns
    ///
    /// Returns the image data as a byte vector on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - HTTP request fails
    /// - Server returns non-success status
    /// - Response body cannot be read
    fn attempt_fetch(&mut self, url_index: Option<usize>) -> Result<Vec<u8>> {
        // Get current URL and advance to next for round-robin
        let current_url = match url_index {
            Some(index) => &self.image_urls[index],
            None => &self.image_urls[self.current_url_index],
        };
        self.current_url_index = (self.current_url_index + 1) % self.image_urls.len();

        let response = reqwest::blocking::get(current_url)?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "HTTP request failed with status: {}",
                response.status()
            ));
        }

        let data = response.bytes()?;
        Ok(data.to_vec())
    }

    /// Get the current retry count.
    ///
    /// Returns the number of failed attempts since the last successful fetch.
    pub fn get_retry_count(&self) -> u32 {
        self.retry_count
    }

    /// Check if a disconnect alert has been sent.
    ///
    /// Returns true if a system offline alert has been sent and we're currently
    /// in a disconnected state.
    pub fn is_disconnect_alert_sent(&self) -> bool {
        self.disconnect_alert_sent
    }

    /// Reset the internal state.
    ///
    /// Resets retry count, alert status, and URL index. Useful for testing or when
    /// manually recovering from error states.
    pub fn reset_state(&mut self) {
        self.retry_count = 0;
        self.disconnect_alert_sent = false;
        self.current_url_index = 0;
    }

    /// Get the configured image URLs.
    pub fn get_image_urls(&self) -> &[String] {
        &self.image_urls
    }

    /// Get the current image URL being used.
    pub fn get_current_image_url(&self) -> &str {
        &self.image_urls[self.current_url_index]
    }

    /// Get all image URLs as a comma-separated string.
    pub fn get_image_urls_string(&self) -> String {
        self.image_urls.join(", ")
    }

    /// Get the maximum retry count.
    pub fn get_max_retries(&self) -> u32 {
        self.max_retries
    }

    /// Convert fetched image bytes directly to a darknet Image.
    ///
    /// This method bypasses disk I/O by loading the image data directly
    /// from memory and converting it to a darknet Image object.
    ///
    /// # Arguments
    ///
    /// * `image_data` - Raw image bytes (JPEG, PNG, etc.)
    ///
    /// # Returns
    ///
    /// Returns a darknet Image object ready for inference.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Image format is not supported
    /// - Image data is corrupted
    /// - Memory allocation fails
    ///
    pub fn bytes_to_darknet_image(image_data: &[u8]) -> Result<Image> {
        // Load image from memory using the image crate
        let dynamic_image: image::DynamicImage = image::load_from_memory(image_data)?;

        // Convert DynamicImage to darknet Image using the explicit From impl
        // Use the fully qualified type to ensure correct trait resolution
        let darknet_image = Image::from(dynamic_image);

        Ok(darknet_image)
    }

    /// Annotate an image with detection boxes and labels.
    ///
    /// This function draws yellow bounding boxes around detected print failures
    /// and adds labels with confidence scores.
    ///
    /// # Arguments
    ///
    /// * `image_data` - Raw image bytes
    /// * `detections` - Vector of detection results to draw
    /// * `image_width` - Width of the original image
    /// * `image_height` - Height of the original image
    ///
    /// # Returns
    ///
    /// Annotated image as JPEG bytes
    pub fn annotate_image_with_detections(
        image_data: &[u8],
        detections: &[Detection],
        image_width: u32,
        image_height: u32,
    ) -> Result<Vec<u8>> {
        // Load the image from memory
        let dynamic_image = image::load_from_memory(image_data)?;

        // Convert to RGB if it's not already
        let mut rgb_image = dynamic_image.to_rgb8();

        // Define yellow color for bounding boxes
        let yellow = image::Rgb([255, 255, 0]);

        // Draw detection boxes
        for detection in detections {
            // Convert darknet coordinates (center + size) to pixel coordinates (top-left + size)
            let center_x = detection.center_x() * image_width as f32;
            let center_y = detection.center_y() * image_height as f32;
            let width = detection.width() * image_width as f32;
            let height = detection.height() * image_height as f32;

            // Calculate top-left corner
            let x = (center_x - width / 2.0).max(0.0) as i32;
            let y = (center_y - height / 2.0).max(0.0) as i32;
            let w = width as u32;
            let h = height as u32;

            // Draw the bounding box with thick yellow border
            for thickness in 0..3 {
                let thick_x = x - thickness;
                let thick_y = y - thickness;
                let thick_w = w + 2 * thickness as u32;
                let thick_h = h + 2 * thickness as u32;

                if thick_x >= 0 && thick_y >= 0 {
                    let rect = Rect::at(thick_x, thick_y).of_size(
                        thick_w.min(image_width.saturating_sub(thick_x as u32)),
                        thick_h.min(image_height.saturating_sub(thick_y as u32)),
                    );
                    draw_hollow_rect_mut(&mut rgb_image, rect, yellow);
                }
            }
        }

        // Convert back to dynamic image and encode as JPEG
        let dynamic_annotated = image::DynamicImage::ImageRgb8(rgb_image);
        let mut buffer = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buffer);

        dynamic_annotated.write_to(&mut cursor, image::ImageFormat::Jpeg)?;

        Ok(buffer)
    }
    /// Apply image transformations (e.g., flipping) to the fetched image.
    ///
    /// # Arguments
    ///
    /// * `image_data` - Raw image bytes
    /// * `flip_vertical` - Whether to flip the image vertically
    ///
    /// # Returns
    ///
    /// Transformed image as bytes in the same format as the input
    pub fn apply_image_transformations(image_data: &[u8], flip_vertical: bool) -> Result<Vec<u8>> {
        if !flip_vertical {
            // No transformations needed, return original data
            return Ok(image_data.to_vec());
        }

        // Load the image from memory
        let dynamic_image = image::load_from_memory(image_data)?;

        // Apply vertical flip
        let flipped_image = dynamic_image.flipv();

        // Encode back to the same format (determine format from input)
        let mut buffer = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buffer);

        // Try to preserve the original format
        let format = image::guess_format(image_data).unwrap_or(image::ImageFormat::Jpeg);
        flipped_image.write_to(&mut cursor, format)?;

        Ok(buffer)
    }
}

/// Types of alerts that can be triggered by the image fetcher.
pub enum AlertType {
    /// System has gone offline due to repeated fetch failures.
    SystemOffline,
    /// System has recovered and is back online.
    SystemRecovery,
}
