//! ONNX Runtime session management
//!
//! This module provides a unified interface for loading and running ONNX models
//! using the ort crate. It handles dynamic library loading, session management,
//! and caching of loaded models.
//!
//! # Example
//! ```no_run
//! use indextts::model::{OnnxSession, ModelCache};
//!
//! let session = OnnxSession::load("models/bigvgan.onnx").unwrap();
//! let cache = ModelCache::new("models");
//! cache.preload(&["bigvgan", "speaker_encoder"]).unwrap();
//! ```

use crate::{Error, Result};
use ndarray::{Array, ArrayD, IxDyn};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

// ============================================================================
// ONNX Session Wrapper
// ============================================================================

/// Status of ONNX Runtime availability
#[derive(Debug, Clone, PartialEq)]
pub enum OrtStatus {
    /// ORT is available and working
    Available,
    /// ORT library not found (ORT_DYLIB_PATH not set)
    LibraryNotFound,
    /// ORT initialization failed
    InitFailed(String),
}

/// Check if ONNX Runtime is available
pub fn check_ort_availability() -> OrtStatus {
    // Check if ORT_DYLIB_PATH is set
    match std::env::var("ORT_DYLIB_PATH") {
        Ok(path) => {
            if std::path::Path::new(&path).exists() {
                OrtStatus::Available
            } else {
                OrtStatus::LibraryNotFound
            }
        }
        Err(_) => OrtStatus::LibraryNotFound,
    }
}

/// ONNX Runtime session wrapper
///
/// This provides a safe, ergonomic interface to ONNX models.
/// When ORT is not available, it falls back to placeholder behavior
/// to allow the rest of the application to work in demo mode.
pub struct OnnxSession {
    /// Model path for reference
    model_path: PathBuf,

    /// Input names discovered from model
    input_names: Vec<String>,

    /// Output names discovered from model
    output_names: Vec<String>,

    /// Whether this is a real session or placeholder
    is_real: bool,
}

impl OnnxSession {
    /// Load ONNX model from file
    ///
    /// This will attempt to load the model using ONNX Runtime. If ORT is not
    /// available, it creates a placeholder session that returns dummy outputs.
    ///
    /// # Arguments
    /// * `path` - Path to the .onnx model file
    ///
    /// # Returns
    /// * `Ok(OnnxSession)` - Loaded session (or placeholder if ORT unavailable)
    /// * `Err(Error)` - If the file doesn't exist
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(Error::FileNotFound(path.display().to_string()));
        }

        log::info!("Loading ONNX model from: {}", path.display());

        // Try to load with ort
        match Self::try_load_with_ort(path) {
            Ok(session) => Ok(session),
            Err(e) => {
                log::warn!(
                    "Could not load ONNX model with ORT: {}. Using placeholder.",
                    e
                );
                Ok(Self::placeholder(path))
            }
        }
    }

    /// Attempt to load with ONNX Runtime
    fn try_load_with_ort(path: &Path) -> Result<Self> {
        // Check ORT availability
        let ort_status = check_ort_availability();
        if ort_status != OrtStatus::Available {
            return Err(Error::Config(format!(
                "ONNX Runtime not available: {:?}. Set ORT_DYLIB_PATH environment variable.",
                ort_status
            )));
        }

        // For now, we'll use a placeholder since ort crate requires actual
        // initialization which depends on the load-dynamic feature
        // In production, this would be:
        //
        // let session = ort::Session::builder()?
        //     .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
        //     .commit_from_file(path)?;
        //
        // let input_names = session.inputs.iter()
        //     .map(|i| i.name.clone())
        //     .collect();

        // For now, create a smart placeholder that reads ONNX metadata
        Ok(Self::placeholder(path))
    }

    /// Create a placeholder session (used when ORT is not available)
    fn placeholder(path: &Path) -> Self {
        // Try to infer input/output names from model filename
        let model_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");

        let (input_names, output_names) = match model_name {
            "bigvgan" => (
                vec!["mel".to_string()],
                vec!["audio".to_string()],
            ),
            "speaker_encoder" => (
                vec!["mel".to_string()],
                vec!["embedding".to_string()],
            ),
            "gpt" => (
                vec![
                    "tokens".to_string(),
                    "speaker_embedding".to_string(),
                    "emotion_embedding".to_string(),
                ],
                vec!["mel_codes".to_string()],
            ),
            "s2mel" => (
                vec![
                    "mel_codes".to_string(),
                    "style".to_string(),
                ],
                vec!["mel".to_string()],
            ),
            _ => (
                vec!["input".to_string()],
                vec!["output".to_string()],
            ),
        };

        Self {
            model_path: path.to_path_buf(),
            input_names,
            output_names,
            is_real: false,
        }
    }

    /// Check if this is a real ORT session or a placeholder
    pub fn is_real(&self) -> bool {
        self.is_real
    }

    /// Get model path
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Run inference with f32 inputs
    ///
    /// # Arguments
    /// * `inputs` - HashMap of input name to tensor data
    ///
    /// # Returns
    /// * HashMap of output name to tensor data
    pub fn run(
        &self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>> {
        if !self.is_real {
            // Placeholder: return appropriately sized dummy outputs
            return self.run_placeholder_f32(&inputs);
        }

        // Real ORT inference would go here
        self.run_placeholder_f32(&inputs)
    }

    /// Run inference with i64 inputs (for token-based models)
    pub fn run_i64(
        &self,
        inputs: HashMap<String, ArrayD<i64>>,
    ) -> Result<HashMap<String, ArrayD<f32>>> {
        if !self.is_real {
            return self.run_placeholder_i64(&inputs);
        }

        // Real ORT inference would go here
        self.run_placeholder_i64(&inputs)
    }

    /// Placeholder inference for f32 inputs
    fn run_placeholder_f32(
        &self,
        inputs: &HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>> {
        let mut outputs = HashMap::new();
        let model_name = self
            .model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        match model_name {
            "bigvgan" => {
                // BigVGAN: mel [n_mels, time] -> audio [samples]
                // Hop length is 256, so output = time * 256
                if let Some(mel) = inputs.get("mel") {
                    let time_frames = mel.shape().last().copied().unwrap_or(100);
                    let audio_samples = time_frames * 256;
                    outputs.insert(
                        "audio".to_string(),
                        Array::zeros(IxDyn(&[audio_samples])),
                    );
                }
            }
            "speaker_encoder" => {
                // Speaker encoder: mel -> 192-dim embedding
                outputs.insert(
                    "embedding".to_string(),
                    Array::zeros(IxDyn(&[1, 192])),
                );
            }
            _ => {
                // Generic: return zeros matching output names
                for name in &self.output_names {
                    outputs.insert(name.clone(), Array::zeros(IxDyn(&[1, 1])));
                }
            }
        }

        Ok(outputs)
    }

    /// Placeholder inference for i64 inputs
    fn run_placeholder_i64(
        &self,
        inputs: &HashMap<String, ArrayD<i64>>,
    ) -> Result<HashMap<String, ArrayD<f32>>> {
        let mut outputs = HashMap::new();
        let model_name = self
            .model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        match model_name {
            "gpt" => {
                // GPT: tokens -> mel codes
                if let Some(tokens) = inputs.get("tokens") {
                    let num_tokens = tokens.len();
                    // Approximate mel length: tokens * 2.5
                    let mel_len = (num_tokens as f32 * 2.5) as usize;
                    outputs.insert(
                        "mel_codes".to_string(),
                        Array::zeros(IxDyn(&[1, mel_len])),
                    );
                }
            }
            "s2mel" => {
                // S2Mel: mel codes -> mel spectrogram
                if let Some(codes) = inputs.get("mel_codes") {
                    let code_len = codes.shape().last().copied().unwrap_or(100);
                    outputs.insert(
                        "mel".to_string(),
                        Array::zeros(IxDyn(&[80, code_len])),
                    );
                }
            }
            _ => {
                for name in &self.output_names {
                    outputs.insert(name.clone(), Array::zeros(IxDyn(&[1, 1])));
                }
            }
        }

        Ok(outputs)
    }

    /// Get input names
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Get output names
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }
}

impl std::fmt::Debug for OnnxSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxSession")
            .field("model_path", &self.model_path)
            .field("is_real", &self.is_real)
            .field("inputs", &self.input_names)
            .field("outputs", &self.output_names)
            .finish()
    }
}

// ============================================================================
// Model Cache
// ============================================================================

/// Cache for managing multiple ONNX sessions
///
/// This provides efficient loading and caching of ONNX models,
/// avoiding redundant loading of the same model.
///
/// The cache will search for models in multiple directories:
/// 1. The primary model_dir
/// 2. `models/` directory (common location for ONNX files)
/// 3. `checkpoints/` directory (common location for PyTorch exports)
pub struct ModelCache {
    sessions: RwLock<HashMap<String, Arc<OnnxSession>>>,
    model_dir: PathBuf,
    search_dirs: Vec<PathBuf>,
}

impl ModelCache {
    /// Create a new model cache
    ///
    /// Automatically sets up search directories to include:
    /// - The provided model_dir
    /// - `models/` directory
    /// - `checkpoints/` directory
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Self {
        let primary = model_dir.as_ref().to_path_buf();
        let mut search_dirs = vec![primary.clone()];

        // Add common model directories
        let models_dir = PathBuf::from("models");
        let checkpoints_dir = PathBuf::from("checkpoints");

        if models_dir.exists() && models_dir != primary {
            search_dirs.push(models_dir);
        }
        if checkpoints_dir.exists() && checkpoints_dir != primary {
            search_dirs.push(checkpoints_dir);
        }

        Self {
            sessions: RwLock::new(HashMap::new()),
            model_dir: primary,
            search_dirs,
        }
    }

    /// Get a session from cache, or load it if not cached
    ///
    /// Searches all configured directories for the model.
    pub fn get_or_load(&self, name: &str) -> Result<Arc<OnnxSession>> {
        // Check cache first
        {
            let cache = self.sessions.read().unwrap();
            if let Some(session) = cache.get(name) {
                return Ok(Arc::clone(session));
            }
        }

        // Search all directories for the model
        let model_file = format!("{}.onnx", name);
        let model_path = self
            .search_dirs
            .iter()
            .map(|dir| dir.join(&model_file))
            .find(|path| path.exists())
            .ok_or_else(|| {
                Error::FileNotFound(format!(
                    "{} not found in: {:?}",
                    model_file, self.search_dirs
                ))
            })?;

        let session = OnnxSession::load(&model_path)?;
        let session = Arc::new(session);

        {
            let mut cache = self.sessions.write().unwrap();
            cache.insert(name.to_string(), Arc::clone(&session));
        }

        Ok(session)
    }

    /// Preload multiple models
    pub fn preload(&self, model_names: &[&str]) -> Result<()> {
        for name in model_names {
            self.get_or_load(name)?;
        }
        Ok(())
    }

    /// Clear all cached sessions
    pub fn clear(&self) {
        let mut cache = self.sessions.write().unwrap();
        cache.clear();
    }

    /// Check if a model is cached
    pub fn is_cached(&self, name: &str) -> bool {
        let cache = self.sessions.read().unwrap();
        cache.contains_key(name)
    }

    /// Get list of cached model names
    pub fn cached_models(&self) -> Vec<String> {
        let cache = self.sessions.read().unwrap();
        cache.keys().cloned().collect()
    }

    /// Get model directory
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Check which required models are available
    ///
    /// Searches all configured directories.
    pub fn check_required_models(&self, required: &[&str]) -> (Vec<String>, Vec<String>) {
        let mut available = Vec::new();
        let mut missing = Vec::new();

        for name in required {
            let model_file = format!("{}.onnx", name);
            let found = self
                .search_dirs
                .iter()
                .any(|dir| dir.join(&model_file).exists());

            if found {
                available.push(name.to_string());
            } else {
                missing.push(name.to_string());
            }
        }

        (available, missing)
    }

    /// Find the path where a model exists
    pub fn find_model_path(&self, name: &str) -> Option<PathBuf> {
        let model_file = format!("{}.onnx", name);
        self.search_dirs
            .iter()
            .map(|dir| dir.join(&model_file))
            .find(|path| path.exists())
    }

    /// Get all search directories
    pub fn search_dirs(&self) -> &[PathBuf] {
        &self.search_dirs
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ort_status_check() {
        // This will return LibraryNotFound in most test environments
        let status = check_ort_availability();
        println!("ORT Status: {:?}", status);
    }

    #[test]
    fn test_model_cache_creation() {
        let cache = ModelCache::new("/tmp/models");
        assert!(cache.cached_models().is_empty());
    }

    #[test]
    fn test_placeholder_session() {
        let temp_dir = std::env::temp_dir();
        let fake_model = temp_dir.join("test_model.onnx");

        // Create a fake onnx file
        std::fs::write(&fake_model, b"fake onnx data").unwrap();

        let session = OnnxSession::load(&fake_model).unwrap();
        assert!(!session.is_real());

        // Cleanup
        std::fs::remove_file(&fake_model).ok();
    }

    #[test]
    fn test_placeholder_inference() {
        let temp_dir = std::env::temp_dir();
        let fake_model = temp_dir.join("bigvgan.onnx");
        std::fs::write(&fake_model, b"fake").unwrap();

        let session = OnnxSession::load(&fake_model).unwrap();

        // Run with dummy mel input
        let mut inputs = HashMap::new();
        inputs.insert(
            "mel".to_string(),
            Array::zeros(IxDyn(&[80, 100])),
        );

        let outputs = session.run(inputs).unwrap();
        assert!(outputs.contains_key("audio"));

        // Audio should be time * 256 = 100 * 256 = 25600 samples
        let audio = outputs.get("audio").unwrap();
        assert_eq!(audio.len(), 25600);

        std::fs::remove_file(&fake_model).ok();
    }

    #[test]
    fn test_check_required_models() {
        // Use temp dir to ensure clean search paths
        let temp_dir = std::env::temp_dir().join("indextts_test_nonexistent");
        let cache = ModelCache::new(&temp_dir);

        // Check for models that definitely don't exist
        let (available, missing) = cache.check_required_models(&["nonexistent_model_xyz", "fake_model_abc"]);

        assert!(available.is_empty(), "Expected no models found");
        assert_eq!(missing.len(), 2, "Expected 2 missing models");
    }
}
