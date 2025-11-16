//! Error types for IndexTTS

use thiserror::Error;

/// Main error type for IndexTTS
#[derive(Error, Debug)]
pub enum Error {
    #[error("Audio processing error: {0}")]
    Audio(String),

    #[error("Text processing error: {0}")]
    Text(String),

    #[error("Model inference error: {0}")]
    Model(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error("ONNX Runtime error: {0}")]
    Onnx(String),

    #[error("Tokenization error: {0}")]
    Tokenization(String),

    #[error("Model loading error: {0}")]
    ModelLoading(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Vocoder error: {0}")]
    Vocoder(String),

    #[error("Unsupported operation: {0}")]
    Unsupported(String),

    #[error("Download error: {0}")]
    Download(String),

    #[error("Shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },
}

/// Result type for IndexTTS operations
pub type Result<T> = std::result::Result<T, Error>;

impl From<serde_yaml::Error> for Error {
    fn from(err: serde_yaml::Error) -> Self {
        Error::Config(err.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Config(err.to_string())
    }
}

impl From<hound::Error> for Error {
    fn from(err: hound::Error) -> Self {
        Error::Audio(err.to_string())
    }
}

impl From<ndarray::ShapeError> for Error {
    fn from(err: ndarray::ShapeError) -> Self {
        Error::ShapeMismatch {
            expected: "valid shape".into(),
            actual: err.to_string(),
        }
    }
}

impl From<regex::Error> for Error {
    fn from(err: regex::Error) -> Self {
        Error::Text(err.to_string())
    }
}
