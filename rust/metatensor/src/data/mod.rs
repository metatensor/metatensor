mod origin;

mod array;
pub use self::array::Array;

mod empty;
pub use self::empty::EmptyArray;

mod ndarray_array;

mod external;
pub use self::external::MtsArray;

mod array_ref;
pub use self::array_ref::{ArrayRef, ArrayRefMut};

pub use dlpk::sys::{DLDataType, DLDataTypeCode};
pub use dlpk::sys::{DLDeviceType, DLDevice};
pub use dlpk::sys::DLPackVersion;
pub use dlpk::DLPackTensor;

pub use metatensor_sys::mts_data_movement_t;

#[cfg(test)]
mod tests {
    use ndarray::Array;
    use crate::c_api::mts_data_movement_t;

    use super::*;
    use super::origin::get_data_origin;

    #[test]
    fn shape() {
        let mut array = MtsArray::from(Array::from_elem(vec![3, 4, 2], 1.0));

        assert_eq!(array.shape().unwrap(), [3, 4, 2]);
        array.reshape(&[12, 2]).unwrap();
        assert_eq!(array.shape().unwrap(), [12, 2]);
        assert_eq!(*array.as_ndarray::<f64>(), Array::from_elem(vec![12, 2], 1.0));

        array.swap_axes(0, 1).unwrap();
        assert_eq!(array.shape().unwrap(), [2, 12]);

        let array_ref = array.as_ref();
        assert_eq!(array_ref.shape().unwrap(), [2, 12]);
        assert_eq!(*array_ref.as_ndarray_lock::<f64>().read().unwrap(), Array::from_elem(vec![2, 12], 1.0));

        let mut array_ref_mut = array.as_mut();
        assert_eq!(array_ref_mut.shape().unwrap(), [2, 12]);

        array_ref_mut.reshape(&[6, 4]).unwrap();
        assert_eq!(array_ref_mut.shape().unwrap(), [6, 4]);
        assert_eq!(*array_ref_mut.as_ndarray_lock::<f64>().read().unwrap(), Array::from_elem(vec![6, 4], 1.0));
    }

    #[test]
    fn create() {
        let array = MtsArray::from(Array::from_elem(vec![4, 2], 1.0));

        assert_eq!(get_data_origin(array.origin().unwrap()).unwrap(), "RustArray");
        assert_eq!(*array.as_ndarray::<f64>(), Array::from_elem(vec![4, 2], 1.0));

        let fill_value = MtsArray::from(Array::from_elem(vec![], 42.0));
        let other = array.create(&[5, 3, 7, 12], fill_value.as_ref()).unwrap();
        assert_eq!(other.shape().unwrap(), [5, 3, 7, 12]);
        assert_eq!(get_data_origin(other.origin().unwrap()).unwrap(), "RustArray");
        assert_eq!(*other.as_ndarray::<f64>(), Array::from_elem(vec![5, 3, 7, 12], 42.0));
    }

    #[test]
    fn move_data() {
        let array = MtsArray::from(Array::from_elem(vec![3, 2, 2, 4], 1.0));

        let fill_value = MtsArray::from(Array::from_elem(vec![], 0.0));
        let mut other = array.create(&[1, 2, 2, 8], fill_value.as_ref()).unwrap();
        assert_eq!(*other.as_ndarray::<f64>(), Array::from_elem(vec![1, 2, 2, 8], 0.0));

        let mapping = mts_data_movement_t {
            sample_in: 1,
            sample_out: 0,
            properties_start_in: 0,
            properties_start_out: 2,
            properties_length: 4,
        };
        other.move_data(&array, &[mapping]).unwrap();
        let expected = Array::from_shape_vec(vec![1, 2, 2, 8], vec![
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
        ]).unwrap();
        assert_eq!(*other.as_ndarray::<f64>(), expected);
    }
}
