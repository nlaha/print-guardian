//! Print Guardian - AI-powered 3D print failure detection system.
//!
//! This library provides components for monitoring 3D printer cameras in real-time
//! using machine learning to detect print failures and automatically respond to them.
//!
//! # Core Components
//!
//! * [`config`] - Configuration management and command-line arguments
//! * [`alerts`] - Discord webhook notification system
//! * [`printer`] - Moonraker API integration for printer control
//! * [`image_fetcher`] - Robust image downloading with retry logic
//! * [`detector`] - AI-powered print failure detection using YOLO/Darknet
//! * [`error`] - Comprehensive error types and handling
//!

pub mod alerts;
pub mod config;
pub mod detector;
pub mod error;
pub mod image_fetcher;
pub mod printer;

// Re-export commonly used types for convenience
pub use alerts::AlertService;
pub use config::Config;
pub use detector::{Detection, FailureDetector};
pub use error::PrintGuardianError;
pub use image_fetcher::ImageFetcher;
pub use printer::PrinterService;
