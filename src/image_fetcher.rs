use anyhow::Result;
use log::{error, info, warn};
use std::{thread, time::Duration};

/// Image fetching service with retry logic and error handling.
///
/// This service handles downloading images from camera endpoints with robust
/// retry logic, timeout handling, and status tracking for reliable operation
/// in network-unstable environments.
pub struct ImageFetcher {
    image_url: String,
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
    /// * `image_url` - URL to fetch images from
    /// * `max_retries` - Maximum number of retry attempts before giving up
    /// * `retry_delay_seconds` - Delay between retry attempts
    ///
    /// # Examples
    ///
    /// ```rust
    /// let fetcher = ImageFetcher::new(
    ///     "http://camera.local/image.jpg".to_string(),
    ///     15,
    ///     15
    /// );
    /// ```
    pub fn new(image_url: String, max_retries: u32, retry_delay_seconds: u64) -> Self {
        Self {
            image_url,
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
    /// # Examples
    ///
    /// ```rust
    /// let image_data = fetcher.fetch_with_retry(|alert_type| {
    ///     // Handle alert sending
    ///     Ok(())
    /// })?;
    /// ```
    pub fn fetch_with_retry<F>(&mut self, mut alert_callback: F) -> Result<Vec<u8>>
    where
        F: FnMut(AlertType) -> Result<()>,
    {
        loop {
            match self.attempt_fetch() {
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
    fn attempt_fetch(&self) -> Result<Vec<u8>> {
        let response = reqwest::blocking::get(&self.image_url)?;

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
    /// Resets retry count and alert status. Useful for testing or when
    /// manually recovering from error states.
    pub fn reset_state(&mut self) {
        self.retry_count = 0;
        self.disconnect_alert_sent = false;
    }

    /// Get the configured image URL.
    pub fn get_image_url(&self) -> &str {
        &self.image_url
    }

    /// Get the maximum retry count.
    pub fn get_max_retries(&self) -> u32 {
        self.max_retries
    }
}

/// Types of alerts that can be triggered by the image fetcher.
pub enum AlertType {
    /// System has gone offline due to repeated fetch failures.
    SystemOffline,
    /// System has recovered and is back online.
    SystemRecovery,
}
