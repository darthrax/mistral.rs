//! Runtime overlays for a model's hashed n-gram (Engram) table.
//!
//! An [`EngramOverlay`] maps a global table row index to `head_dim` f32 values.
//! A model that owns an Engram table registers its overlay handle when it loads;
//! a caller in the same process fetches it with [`registered_overlays`] and
//! writes memory into the running model.
//!
//! The table is addressed per head as `offset[head] + row`. The caller is
//! responsible for resolving tokens to row indices (with the model's hash
//! configuration) and for row bounds; this type only stores and validates
//! whole rows.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

/// A set of runtime edits to an Engram table row.
#[derive(Debug, Clone)]
pub struct EngramOverlay {
    head_dim: usize,
    rows: BTreeMap<u64, Vec<f32>>,
}

impl EngramOverlay {
    /// Create an empty overlay for a table whose rows are `head_dim` wide.
    pub fn new(head_dim: usize) -> Self {
        Self {
            head_dim,
            rows: BTreeMap::new(),
        }
    }

    /// The row width this overlay accepts.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Insert or replace a row, returning the previous values if any.
    ///
    /// # Errors
    ///
    /// Returns [`EngramError::WrongWidth`] if `values` is not `head_dim` long.
    pub fn set(&mut self, row: u64, values: Vec<f32>) -> Result<Option<Vec<f32>>, EngramError> {
        if values.len() != self.head_dim {
            return Err(EngramError::WrongWidth {
                expected: self.head_dim,
                actual: values.len(),
            });
        }
        Ok(self.rows.insert(row, values))
    }

    /// Remove a row, returning its values if it was present.
    pub fn remove(&mut self, row: u64) -> Option<Vec<f32>> {
        self.rows.remove(&row)
    }

    /// Read a row's values.
    pub fn get(&self, row: u64) -> Option<&[f32]> {
        self.rows.get(&row).map(Vec::as_slice)
    }

    /// Number of overridden rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether no rows are overridden.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Remove every override.
    pub fn clear(&mut self) {
        self.rows.clear();
    }
}

/// Why an overlay edit was rejected.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EngramError {
    /// The supplied row had the wrong number of values.
    #[error("engram overlay row must have {expected} values, got {actual}")]
    WrongWidth {
        /// The required width.
        expected: usize,
        /// The supplied width.
        actual: usize,
    },
}

/// A shared, mutable handle to an overlay.
pub type EngramOverlayHandle = Arc<RwLock<EngramOverlay>>;

static REGISTRY: OnceLock<Mutex<Vec<EngramOverlayHandle>>> = OnceLock::new();

/// Register an overlay handle; called by a model when it loads its Engram table.
pub(crate) fn register_overlay(handle: EngramOverlayHandle) {
    let registry = REGISTRY.get_or_init(|| Mutex::new(Vec::new()));
    if let Ok(mut handles) = registry.lock() {
        handles.push(handle);
    }
}

/// Every overlay registered by models loaded in this process.
///
/// One model per process is the supported configuration; the list exists so a
/// caller can be explicit if more than one is ever loaded.
pub fn registered_overlays() -> Vec<EngramOverlayHandle> {
    REGISTRY
        .get()
        .and_then(|registry| registry.lock().ok().map(|handles| handles.clone()))
        .unwrap_or_default()
}

/// Forget every registered overlay. Intended for tests.
pub fn clear_registered_overlays() {
    if let Some(registry) = REGISTRY.get() {
        if let Ok(mut handles) = registry.lock() {
            handles.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_sets_gets_and_removes_rows() {
        let mut overlay = EngramOverlay::new(3);
        assert!(overlay.is_empty());
        assert_eq!(overlay.set(7, vec![1.0, 2.0, 3.0]).unwrap(), None);
        assert_eq!(overlay.get(7), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(
            overlay.set(7, vec![4.0, 5.0, 6.0]).unwrap(),
            Some(vec![1.0, 2.0, 3.0])
        );
        assert_eq!(overlay.len(), 1);
        assert_eq!(overlay.remove(7), Some(vec![4.0, 5.0, 6.0]));
        assert!(overlay.is_empty());
    }

    #[test]
    fn overlay_rejects_a_row_of_the_wrong_width() {
        let mut overlay = EngramOverlay::new(3);
        assert_eq!(
            overlay.set(1, vec![1.0, 2.0]),
            Err(EngramError::WrongWidth {
                expected: 3,
                actual: 2
            })
        );
    }

    #[test]
    fn registry_starts_empty_and_registers_handles() {
        clear_registered_overlays();
        assert!(registered_overlays().is_empty());

        let handle = Arc::new(RwLock::new(EngramOverlay::new(2)));
        register_overlay(handle.clone());

        let handles = registered_overlays();
        assert_eq!(handles.len(), 1);
        assert!(Arc::ptr_eq(&handles[0], &handle));

        clear_registered_overlays();
        assert!(registered_overlays().is_empty());
    }
}
