use dlpk::sys::{DLDevice, DLPackVersion, DLDataType};
use dlpk::{DLPackTensor};

use crate::errors::Error;
use crate::c_api::mts_data_movement_t;

use super::{Array, MtsArray};


/// An implementation of the [`Array`] trait without any data.
///
/// This only tracks the shape of the array.
#[derive(Debug, Clone)]
pub struct EmptyArray {
    shape: Vec<usize>,
}

impl EmptyArray {
    /// Create a new `EmptyArray` with the given shape.
    pub fn new(shape: Vec<usize>) -> EmptyArray {
        EmptyArray { shape }
    }
}

impl Array for EmptyArray {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn create(&self, shape: &[usize], _fill_value: MtsArray) -> Box<dyn Array> {
        Box::new(EmptyArray { shape: shape.to_vec() })
    }

    fn copy(&self) -> Box<dyn Array> {
        Box::new(EmptyArray { shape: self.shape.clone() })
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn reshape(&mut self, shape: &[usize]) {
        self.shape = shape.to_vec();
    }

    fn swap_axes(&mut self, axis_1: usize, axis_2: usize) {
        self.shape.swap(axis_1, axis_2);
    }

    fn move_data(&mut self, _: &dyn Array, _: &[mts_data_movement_t]) {
        panic!("can not call Array::move_data() for EmptyArray");
    }

    fn device(&self) -> DLDevice {
        DLDevice::cpu()
    }

    fn dtype(&self) -> DLDataType {
        // Default to f64, consistent with metatensor-core's EmptyDataArray
        DLDataType {
            code: dlpk::sys::DLDataTypeCode::kDLFloat,
            bits: 64,
            lanes: 1,
        }
    }

    fn as_dlpack(
        &self,
        _device: DLDevice,
        _stream: Option<i64>,
        _max_version: DLPackVersion
    ) -> Result<DLPackTensor, Error> {
        panic!("can not call Array::as_dlpack() for EmptyArray");
    }
}

#[cfg(test)]
mod tests {
    use dlpk::sys::DLDevice;
    use crate::Array;

    use super::*;

    #[test]
    fn empty_array_device() {
        let array = EmptyArray::new(vec![2, 3]);
        assert_eq!(array.device(), DLDevice::cpu());
    }
}
