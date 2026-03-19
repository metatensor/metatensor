use dlpk::sys::{DLDevice, DLPackVersion, DLDataType};
use dlpk::{DLPackTensor, GetDLPackDataType, DLPackPointerCast};

use crate::errors::Error;
use crate::c_api::mts_data_movement_t;

use super::{Array, MtsArray};

impl<T> Array for ndarray::ArcArray<T, ndarray::IxDyn>
where
    T: 'static + Send + Sync + Clone + Default + GetDLPackDataType + DLPackPointerCast,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn create(&self, shape: &[usize], fill_value: MtsArray) -> Box<dyn Array> {
        let cpu_device = DLDevice::cpu();
        let max_version = DLPackVersion::current();
        let fill_value_dlpack = fill_value.as_dlpack(cpu_device, None, max_version)
            .expect("failed to extract fill_value as DLPack");

        // Validate fill_value shape from the DLPack tensor directly
        assert_eq!(fill_value_dlpack.shape(), [1], "fill_value must have shape (1,)");

        let fill_value_ptr = fill_value_dlpack.data_ptr::<T>().expect("dtype mismatch between array and fill_value");
        let fill_value_scalar = unsafe { std::ptr::read(fill_value_ptr) };

        Box::new(ndarray::ArcArray::from_elem(shape, fill_value_scalar))
    }

    fn copy(&self) -> Box<dyn Array> {
        return Box::new(self.clone());
    }

    fn shape(&self) -> &[usize] {
        return self.shape();
    }

    fn reshape(&mut self, shape: &[usize]) {
        *self = self.to_shape(shape).expect("invalid shape").to_shared();
    }

    fn swap_axes(&mut self, axis_1: usize, axis_2: usize) {
        self.swap_axes(axis_1, axis_2);
    }

    fn move_data(
        &mut self,
        input: &dyn Array,
        movements: &[mts_data_movement_t],
    ) {
        use ndarray::{Axis, Slice};

        let input = input.as_any().downcast_ref::<ndarray::ArcArray<T, ndarray::IxDyn>>().expect("input must be a ndarray of the same type");

        if movements.is_empty() {
            return;
        }

        // Check if we can use the optimized path (all moves have same property structure)
        let first_prop_start_in = movements[0].properties_start_in;
        let first_prop_start_out = movements[0].properties_start_out;
        let first_prop_len = movements[0].properties_length;

        let mut constant_properties = true;
        let mut contiguous_input_samples = true;
        let mut contiguous_output_samples = true;

        for w in movements.windows(2) {
            if w[0].properties_start_in != first_prop_start_in ||
               w[0].properties_start_out != first_prop_start_out ||
               w[0].properties_length != first_prop_len {
                constant_properties = false;
                break;
            }

            if w[1].sample_in != w[0].sample_in + 1 {
                contiguous_input_samples = false;
            }

            if w[1].sample_out != w[0].sample_out + 1 {
                contiguous_output_samples = false;
            }
        }

        if constant_properties {
            let last = movements.last().unwrap();
            if last.properties_start_in != first_prop_start_in ||
               last.properties_start_out != first_prop_start_out ||
               last.properties_length != first_prop_len {
                constant_properties = false;
            }
        }

        let property_axis = self.shape().len() - 1;

        if constant_properties {
            let input_slice_info = Slice::from(first_prop_start_in..(first_prop_start_in + first_prop_len));
            let output_slice_info = Slice::from(first_prop_start_out..(first_prop_start_out + first_prop_len));

            if contiguous_input_samples && contiguous_output_samples {
                let sample_start_in = movements[0].sample_in;
                let sample_start_out = movements[0].sample_out;
                let sample_count = movements.len();

                let input_samples = input.slice_axis(
                    Axis(0),
                    Slice::from(sample_start_in..(sample_start_in + sample_count))
                );
                let mut output_samples = self.slice_axis_mut(
                    Axis(0),
                    Slice::from(sample_start_out..(sample_start_out + sample_count))
                );

                let value = input_samples.slice_axis(Axis(property_axis), input_slice_info);
                let mut output_location = output_samples.slice_axis_mut(Axis(property_axis), output_slice_info);

                output_location.assign(&value);
            } else {
                for move_item in movements {
                    let input_sample = input.index_axis(Axis(0), move_item.sample_in);
                    let mut output_sample = self.index_axis_mut(Axis(0), move_item.sample_out);

                    let value = input_sample.slice_axis(
                        // property_axis - 1 because we are slicing the sample
                        // axis out, so the property axis is now one less
                        Axis(property_axis - 1),
                        input_slice_info
                    );
                    let mut output_location = output_sample.slice_axis_mut(
                        Axis(property_axis - 1),
                        output_slice_info
                    );
                    output_location.assign(&value);
                }
            }
        } else {
            // fallback to the general case
            for move_item in movements {
                let input_sample = input.index_axis(Axis(0), move_item.sample_in);
                let mut output_sample = self.index_axis_mut(Axis(0), move_item.sample_out);

                let value = input_sample.slice_axis(
                    // see above for property_axis - 1 explanation
                    Axis(property_axis - 1),
                    Slice::from(move_item.properties_start_in..(move_item.properties_start_in + move_item.properties_length))
                );
                let mut output_location = output_sample.slice_axis_mut(
                    Axis(property_axis - 1),
                    Slice::from(move_item.properties_start_out..(move_item.properties_start_out + move_item.properties_length))
                );
                output_location.assign(&value);
            }
        }
    }

    fn device(&self) -> DLDevice {
        DLDevice::cpu()
    }

    fn dtype(&self) -> DLDataType {
        T::get_dlpack_data_type()
    }

    fn as_dlpack(
        &self,
        device: DLDevice,
        stream: Option<i64>,
        max_version: DLPackVersion,
    ) -> Result<DLPackTensor, Error> {
        if stream.is_some() {
            // we only support CPU for now
            return Err(Error {
                code: Some(crate::c_api::MTS_INVALID_PARAMETER_ERROR),
                message: "CPU arrays can not be used with a stream".into(),
            });
        }
        let vendored_version = DLPackVersion::current();
        let major_mismatch = max_version.major != vendored_version.major;
        let minor_too_high = max_version.minor < vendored_version.minor;
        if major_mismatch || minor_too_high {
            return Err(Error {
                code: Some(crate::c_api::MTS_INVALID_PARAMETER_ERROR),
                message: format!(
                    "Metatensor supports DLPack version {}.{}. Caller requested incompatible version {}.{}",
                    vendored_version.major, vendored_version.minor, max_version.major, max_version.minor
                ),
            });
        }

        let ndarray_device = DLDevice::cpu();

        if device.device_type != ndarray_device.device_type || device.device_id != ndarray_device.device_id {
            return Err(Error {
                code: Some(crate::c_api::MTS_INVALID_PARAMETER_ERROR),
                message: format!(
                    "Requested DLPack device ({}) does not match array device ({})",
                    device, ndarray_device
                ),
            });
        }

        let tensor: DLPackTensor = self.try_into().map_err(|e| Error {
            code: Some(crate::c_api::MTS_INVALID_PARAMETER_ERROR),
            message: format!("failed to convert ndarray to DLPack: {:?}", e),
        })?;

        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use dlpk::{DLPackPointerCast, GetDLPackDataType, sys::{DLDataTypeCode, DLDevice, DLPackVersion}};
    use crate::MtsArray;

    #[test]
    fn ndarray_as_mts_array() {
        let data = ndarray::ArcArray::<f64, _>::zeros(vec![2, 3, 4]);
        let mts_array = MtsArray::new(data);

        assert_eq!(mts_array.shape().unwrap(), [2, 3, 4]);

        let fill_value = MtsArray::new(ndarray::ArcArray::from_elem(vec![1], 42.0));

        let created = mts_array.create(&[2, 3, 4], fill_value.as_ref()).unwrap();
        assert_eq!(created.shape().unwrap(), [2, 3, 4]);
    }

    #[test]
    fn ndarray_as_mts_array_dlpack() {
        let data = ndarray::ArcArray::<f64, _>::zeros(vec![4, 5, 6]);
        let mts_array = MtsArray::new(data);

        let dl_managed = mts_array.as_dlpack(DLDevice::cpu(), None, DLPackVersion::current()).unwrap();

        assert_eq!(dl_managed.n_dims(), 3);
        assert_eq!(dl_managed.shape(), [4, 5, 6]);

        assert_eq!(dl_managed.dtype().code, DLDataTypeCode::kDLFloat);
        assert_eq!(dl_managed.dtype().bits, 64);
        assert_eq!(dl_managed.dtype().lanes, 1);
    }

    #[test]
    fn ndarray_all_dtypes() {
        fn test_for_dtype<T>(code: DLDataTypeCode, bits: u8) where T: Send + Sync + Clone + Default + GetDLPackDataType + DLPackPointerCast + 'static {
            let data = ndarray::ArcArray::<T, _>::from_elem(vec![2, 2], T::default());
            let mts_array = MtsArray::new(data);

            assert_eq!(mts_array.shape().unwrap(), [2, 2]);

            // Should be able to export as DLPack
            let dl_managed = mts_array.as_dlpack(DLDevice::cpu(), None, DLPackVersion::current()).unwrap();
            assert_eq!(dl_managed.dtype().code, code);
            assert_eq!(dl_managed.dtype().bits, bits);
            assert_eq!(dl_managed.dtype().lanes, 1);


            // And `create` should make an array of the same type (i32)
            let fill_value = MtsArray::new(ndarray::ArcArray::from_elem(vec![1], T::default()));

            let created = mts_array.create(&[1, 1], fill_value.as_ref()).unwrap();
            let dl_managed = created.as_dlpack(DLDevice::cpu(), None, DLPackVersion::current()).unwrap();

            assert_eq!(dl_managed.dtype().code, code);
            assert_eq!(dl_managed.dtype().bits, bits);
            assert_eq!(dl_managed.dtype().lanes, 1);
        }

        test_for_dtype::<bool>(DLDataTypeCode::kDLBool, 8);
        test_for_dtype::<f64>(DLDataTypeCode::kDLFloat, 64);
        test_for_dtype::<f32>(DLDataTypeCode::kDLFloat, 32);
        test_for_dtype::<i8>(DLDataTypeCode::kDLInt, 8);
        test_for_dtype::<i16>(DLDataTypeCode::kDLInt, 16);
        test_for_dtype::<i32>(DLDataTypeCode::kDLInt, 32);
        test_for_dtype::<i64>(DLDataTypeCode::kDLInt, 64);
        test_for_dtype::<u8>(DLDataTypeCode::kDLUInt, 8);
        test_for_dtype::<u16>(DLDataTypeCode::kDLUInt, 16);
        test_for_dtype::<u32>(DLDataTypeCode::kDLUInt, 32);
        test_for_dtype::<u64>(DLDataTypeCode::kDLUInt, 64);
    }

    #[test]
    fn ndarray_device() {
        let data = ndarray::ArcArray::<f64, _>::zeros(vec![2, 3]);
        let mts_array = MtsArray::new(data);

        assert_eq!(mts_array.device().unwrap(), DLDevice::cpu());
    }

    #[test]
    fn as_dlpack_rejects_stream() {
        let data = ndarray::ArcArray::<f64, _>::zeros(vec![2, 3]);
        let mts_array = MtsArray::new(data);
        match mts_array.as_dlpack(DLDevice::cpu(), Some(42), DLPackVersion::current()) {
            Err(e) => assert!(e.message.contains("stream"), "{}", e.message),
            Ok(_) => panic!("expected error for non-null stream"),
        }
    }

    #[test]
    fn as_dlpack_rejects_wrong_device() {
        let data = ndarray::ArcArray::<f64, _>::zeros(vec![2, 3]);
        let mts_array = MtsArray::new(data);
        let cuda = DLDevice {
            device_type: dlpk::sys::DLDeviceType::kDLCUDA,
            device_id: 0,
        };
        match mts_array.as_dlpack(cuda, None, DLPackVersion::current()) {
            Err(e) => assert!(e.message.contains("does not match"), "{}", e.message),
            Ok(_) => panic!("expected error for CUDA device on CPU array"),
        }
    }

    #[test]
    fn as_dlpack_rejects_incompatible_version() {
        let data = ndarray::ArcArray::<f64, _>::zeros(vec![2, 3]);
        let mts_array = MtsArray::new(data);

        let bad_version = DLPackVersion { major: 99, minor: 0 };
        match mts_array.as_dlpack(DLDevice::cpu(), None, bad_version) {
            Err(e) => assert!(e.message.contains("version"), "{}", e.message),
            Ok(_) => panic!("expected error for incompatible DLPack version"),
        }
    }
}
