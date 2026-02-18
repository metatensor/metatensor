mod origin;

mod array_ref;
pub use self::array_ref::{ArrayRef, ArrayRefMut};

mod array;
pub use self::array::Array;
pub use self::array::EmptyArray;


#[cfg(test)]
mod tests {
    use ndarray::ArcArray;
    use crate::c_api::mts_data_movement_t;

    use super::*;
    use super::origin::get_data_origin;

    #[test]
    fn shape() {
        let array = Box::new(ArcArray::from_elem(vec![3, 4, 2], 1.0)) as Box<dyn Array>;
        let mut array = unsafe { ArrayRefMut::new(array.into()) };

        assert_eq!(array.as_raw().shape().unwrap(), [3, 4, 2]);
        array.as_raw_mut().reshape(&[12, 2]).unwrap();
        assert_eq!(array.as_raw().shape().unwrap(), [12, 2]);
        assert_eq!(array.as_array(), ArcArray::from_elem(vec![12, 2], 1.0));

        array.as_raw_mut().swap_axes(0, 1).unwrap();
        assert_eq!(array.as_raw().shape().unwrap(), [2, 12]);
    }

    #[test]
    fn create() {
        let array = Box::new(ArcArray::from_elem(vec![4, 2], 1.0)) as Box<dyn Array>;
        let array = unsafe { ArrayRef::from_raw(array.into()) };

        assert_eq!(get_data_origin(array.as_raw().origin().unwrap()).unwrap(), "rust.Box<dyn Array>");
        assert_eq!(array.as_array(), ArcArray::from_elem(vec![4, 2], 1.0));

        let other = unsafe { ArrayRef::from_raw(array.as_raw().create(&[5, 3, 7, 12]).unwrap()) };
        assert_eq!(other.as_raw().shape().unwrap(), [5, 3, 7, 12]);
        assert_eq!(get_data_origin(other.as_raw().origin().unwrap()).unwrap(), "rust.Box<dyn Array>");
        assert_eq!(other.as_array(), ArcArray::from_elem(vec![5, 3, 7, 12], 0.0));
    }

    #[test]
    fn move_data() {
        let array = ArcArray::from_elem(vec![3, 2, 2, 4], 1.0);
        let array = unsafe { ArrayRefMut::new((Box::new(array) as Box<dyn Array>).into()) };

        let mut other = unsafe { ArrayRefMut::new(array.as_raw().create(&[1, 2, 2, 8]).unwrap()) };
        let expected = ArcArray::from_elem(vec![1, 2, 2, 8], 0.0);
        assert_eq!(other.as_array(), expected);

        let mapping = mts_data_movement_t {
            sample_in: 1,
            sample_out: 0,
            properties_start_in: 0,
            properties_start_out: 2,
            properties_length: 4,
        };
        other.as_raw_mut().move_data(array.as_raw(), &[mapping]).unwrap();
        let expected = ArcArray::from_shape_vec(vec![1, 2, 2, 8], vec![
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
        ]).unwrap();
        assert_eq!(other.as_array(), expected);
    }
}
