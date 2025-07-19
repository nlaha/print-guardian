use std::fmt;

/// Custom error types for the Print Guardian application.
///
/// This module defines specific error types that can occur throughout the
/// application, providing better error handling and more informative
/// error messages for different failure scenarios.

/// Main error type for Print Guardian operations.
#[derive(Debug)]
pub enum PrintGuardianError {
    /// Errors related to image fetching and processing.
    ImageError(ImageError),

    /// Errors related to neural network detection.
    DetectionError(DetectionError),

    /// Errors related to printer control operations.
    PrinterError(PrinterError),

    /// Errors related to alert/notification systems.
    AlertError(AlertError),

    /// Configuration and setup errors.
    ConfigError(ConfigError),

    /// Network and connectivity errors.
    NetworkError(NetworkError),
}

/// Errors specific to image fetching and processing operations.
#[derive(Debug)]
pub enum ImageError {
    /// Failed to download image from remote URL.
    DownloadFailed { url: String, reason: String },

    /// Image file could not be saved to disk.
    SaveFailed { path: String, reason: String },

    /// Image format is not supported or corrupted.
    InvalidFormat { path: String },

    /// Image file could not be opened or read.
    ReadFailed { path: String, reason: String },
}

/// Errors specific to neural network detection operations.
#[derive(Debug)]
pub enum DetectionError {
    /// Model configuration file could not be loaded.
    ModelConfigLoadFailed { path: String, reason: String },

    /// Model weights file could not be loaded.
    WeightsLoadFailed { path: String, reason: String },

    /// Labels file could not be read or parsed.
    LabelsLoadFailed { path: String, reason: String },

    /// Neural network inference failed.
    InferenceFailed { reason: String },

    /// Model weights download failed.
    WeightsDownloadFailed { url: String, reason: String },
}

/// Errors specific to printer control operations.
#[derive(Debug)]
pub enum PrinterError {
    /// Failed to connect to printer API.
    ConnectionFailed { api_url: String, reason: String },

    /// Printer API returned an error response.
    ApiError {
        endpoint: String,
        status: u16,
        message: String,
    },

    /// Printer is in an invalid state for the requested operation.
    InvalidState {
        requested_action: String,
        current_state: String,
    },

    /// Authentication with printer API failed.
    AuthenticationFailed { api_url: String },
}

/// Errors specific to alert and notification systems.
#[derive(Debug)]
pub enum AlertError {
    /// Discord webhook request failed.
    WebhookFailed { reason: String },

    /// Alert message formatting failed.
    MessageFormatError { reason: String },

    /// Invalid webhook URL provided.
    InvalidWebhookUrl { url: String },
}

/// Errors related to configuration and application setup.
#[derive(Debug)]
pub enum ConfigError {
    /// Required environment variable is missing.
    MissingEnvVar { var_name: String },

    /// Configuration file could not be read.
    FileReadError { path: String, reason: String },

    /// Invalid configuration values provided.
    InvalidValue {
        field: String,
        value: String,
        reason: String,
    },

    /// Required file path does not exist.
    MissingFile { path: String },
}

/// Errors related to network connectivity and communication.
#[derive(Debug)]
pub enum NetworkError {
    /// Generic network request failed.
    RequestFailed { url: String, reason: String },

    /// Network timeout occurred.
    Timeout { url: String, timeout_seconds: u64 },

    /// DNS resolution failed.
    DnsError { hostname: String },

    /// SSL/TLS certificate verification failed.
    SslError { url: String, reason: String },
}

// Implement Display trait for user-friendly error messages
impl fmt::Display for PrintGuardianError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrintGuardianError::ImageError(e) => write!(f, "Image error: {}", e),
            PrintGuardianError::DetectionError(e) => write!(f, "Detection error: {}", e),
            PrintGuardianError::PrinterError(e) => write!(f, "Printer error: {}", e),
            PrintGuardianError::AlertError(e) => write!(f, "Alert error: {}", e),
            PrintGuardianError::ConfigError(e) => write!(f, "Configuration error: {}", e),
            PrintGuardianError::NetworkError(e) => write!(f, "Network error: {}", e),
        }
    }
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImageError::DownloadFailed { url, reason } => {
                write!(f, "Failed to download image from '{}': {}", url, reason)
            }
            ImageError::SaveFailed { path, reason } => {
                write!(f, "Failed to save image to '{}': {}", path, reason)
            }
            ImageError::InvalidFormat { path } => {
                write!(f, "Invalid or corrupted image format: {}", path)
            }
            ImageError::ReadFailed { path, reason } => {
                write!(f, "Failed to read image from '{}': {}", path, reason)
            }
        }
    }
}

impl fmt::Display for DetectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DetectionError::ModelConfigLoadFailed { path, reason } => {
                write!(f, "Failed to load model config from '{}': {}", path, reason)
            }
            DetectionError::WeightsLoadFailed { path, reason } => {
                write!(
                    f,
                    "Failed to load model weights from '{}': {}",
                    path, reason
                )
            }
            DetectionError::LabelsLoadFailed { path, reason } => {
                write!(f, "Failed to load labels from '{}': {}", path, reason)
            }
            DetectionError::InferenceFailed { reason } => {
                write!(f, "Neural network inference failed: {}", reason)
            }
            DetectionError::WeightsDownloadFailed { url, reason } => {
                write!(
                    f,
                    "Failed to download model weights from '{}': {}",
                    url, reason
                )
            }
        }
    }
}

impl fmt::Display for PrinterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrinterError::ConnectionFailed { api_url, reason } => {
                write!(
                    f,
                    "Failed to connect to printer at '{}': {}",
                    api_url, reason
                )
            }
            PrinterError::ApiError {
                endpoint,
                status,
                message,
            } => {
                write!(
                    f,
                    "Printer API error at '{}' (HTTP {}): {}",
                    endpoint, status, message
                )
            }
            PrinterError::InvalidState {
                requested_action,
                current_state,
            } => {
                write!(
                    f,
                    "Cannot {} printer in state '{}'",
                    requested_action, current_state
                )
            }
            PrinterError::AuthenticationFailed { api_url } => {
                write!(f, "Authentication failed for printer API at '{}'", api_url)
            }
        }
    }
}

impl fmt::Display for AlertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlertError::WebhookFailed { reason } => {
                write!(f, "Discord webhook request failed: {}", reason)
            }
            AlertError::MessageFormatError { reason } => {
                write!(f, "Failed to format alert message: {}", reason)
            }
            AlertError::InvalidWebhookUrl { url } => {
                write!(f, "Invalid Discord webhook URL: {}", url)
            }
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::MissingEnvVar { var_name } => {
                write!(f, "Required environment variable '{}' is not set", var_name)
            }
            ConfigError::FileReadError { path, reason } => {
                write!(
                    f,
                    "Failed to read configuration file '{}': {}",
                    path, reason
                )
            }
            ConfigError::InvalidValue {
                field,
                value,
                reason,
            } => {
                write!(
                    f,
                    "Invalid value '{}' for field '{}': {}",
                    value, field, reason
                )
            }
            ConfigError::MissingFile { path } => {
                write!(f, "Required file not found: {}", path)
            }
        }
    }
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetworkError::RequestFailed { url, reason } => {
                write!(f, "Network request to '{}' failed: {}", url, reason)
            }
            NetworkError::Timeout {
                url,
                timeout_seconds,
            } => {
                write!(
                    f,
                    "Request to '{}' timed out after {} seconds",
                    url, timeout_seconds
                )
            }
            NetworkError::DnsError { hostname } => {
                write!(f, "DNS resolution failed for hostname '{}'", hostname)
            }
            NetworkError::SslError { url, reason } => {
                write!(f, "SSL/TLS error for '{}': {}", url, reason)
            }
        }
    }
}

// Implement std::error::Error trait
impl std::error::Error for PrintGuardianError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PrintGuardianError::ImageError(e) => Some(e),
            PrintGuardianError::DetectionError(e) => Some(e),
            PrintGuardianError::PrinterError(e) => Some(e),
            PrintGuardianError::AlertError(e) => Some(e),
            PrintGuardianError::ConfigError(e) => Some(e),
            PrintGuardianError::NetworkError(e) => Some(e),
        }
    }
}

impl std::error::Error for ImageError {}
impl std::error::Error for DetectionError {}
impl std::error::Error for PrinterError {}
impl std::error::Error for AlertError {}
impl std::error::Error for ConfigError {}
impl std::error::Error for NetworkError {}

// Conversion traits for easy error propagation
impl From<ImageError> for PrintGuardianError {
    fn from(err: ImageError) -> Self {
        PrintGuardianError::ImageError(err)
    }
}

impl From<DetectionError> for PrintGuardianError {
    fn from(err: DetectionError) -> Self {
        PrintGuardianError::DetectionError(err)
    }
}

impl From<PrinterError> for PrintGuardianError {
    fn from(err: PrinterError) -> Self {
        PrintGuardianError::PrinterError(err)
    }
}

impl From<AlertError> for PrintGuardianError {
    fn from(err: AlertError) -> Self {
        PrintGuardianError::AlertError(err)
    }
}

impl From<ConfigError> for PrintGuardianError {
    fn from(err: ConfigError) -> Self {
        PrintGuardianError::ConfigError(err)
    }
}

impl From<NetworkError> for PrintGuardianError {
    fn from(err: NetworkError) -> Self {
        PrintGuardianError::NetworkError(err)
    }
}
