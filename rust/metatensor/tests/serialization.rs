mod tensor {
    use std::io::Read;

    use metatensor::TensorMap;

    const DATA_PATH: &str = "../../metatensor-core/tests/data.mts";

    #[test]
    fn load_file() {
        let tensor = metatensor::io::load(DATA_PATH).unwrap();
        check_tensor(&tensor);

        let tensor = TensorMap::load(DATA_PATH).unwrap();
        check_tensor(&tensor);
    }

    #[test]
    fn load_buffer() {
        let mut file = std::fs::File::open(DATA_PATH).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let tensor = metatensor::io::load_buffer(&buffer).unwrap();
        check_tensor(&tensor);

        let tensor = TensorMap::load_buffer(&buffer).unwrap();
        check_tensor(&tensor);
    }

    #[test]
    fn save() {
        let mut file = std::fs::File::open(DATA_PATH).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let tensor = metatensor::io::load_buffer(&buffer).unwrap();

        let temdir = tempfile::tempdir().unwrap();
        let mut tempath = temdir.path().to_path_buf();
        tempath.push("tensor.mts");
        tensor.save(&tempath).unwrap();

        let mut file = std::fs::File::open(tempath).unwrap();
        let mut saved = Vec::new();
        file.read_to_end(&mut saved).unwrap();

        assert_eq!(buffer.len(), saved.len());
        assert_eq!(buffer, saved);
    }

    #[test]
    fn save_buffer() {
        let mut file = std::fs::File::open(DATA_PATH).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let tensor = metatensor::io::load_buffer(&buffer).unwrap();

        let mut saved = Vec::new();
        metatensor::io::save_buffer(&tensor, &mut saved).unwrap();
        assert_eq!(buffer.len(), saved.len());
        assert_eq!(buffer, saved);

        saved.clear();
        tensor.save_buffer(&mut saved).unwrap();
        assert_eq!(buffer.len(), saved.len());
        assert_eq!(buffer, saved);
    }

    fn check_tensor(tensor: &TensorMap) {
        assert_eq!(tensor.keys().names(), ["o3_lambda", "o3_sigma", "center_type", "neighbor_type"]);
        assert_eq!(tensor.keys().count(), 27);

        let block = tensor.block_by_id(13);

        assert_eq!(block.values().device().unwrap(), dlpk::sys::DLDevice::cpu());
        assert_eq!(block.values().shape().unwrap(), [9, 3, 3]);
        assert_eq!(block.samples().names(), ["system", "atom"]);
        assert_eq!(block.components().len(), 1);
        assert_eq!(block.components()[0].names(), ["o3_mu"]);
        assert_eq!(block.properties().names(), ["n"]);

        assert_eq!(block.gradient_list(), ["positions"]);
        let gradient = block.gradient("positions").unwrap();
        assert_eq!(block.values().device().unwrap(), dlpk::sys::DLDevice::cpu());
        assert_eq!(gradient.values().shape().unwrap(), [27, 3, 3, 3]);
        assert_eq!(gradient.samples().names(), ["sample", "system", "atom"]);
        assert_eq!(gradient.components().len(), 2);
        assert_eq!(gradient.components()[0].names(), ["xyz"]);
        assert_eq!(gradient.components()[1].names(), ["o3_mu"]);
        assert_eq!(gradient.properties().names(), ["n"]);
    }
}

mod block {
    use std::io::Read;
    use metatensor::{TensorBlock, TensorBlockRef};

    const DATA_PATH: &str = "../../metatensor-core/tests/block.mts";

    #[test]
    fn load_file() {
        let block = metatensor::io::load_block(DATA_PATH).unwrap();
        check_block(block.as_ref());

        let block = TensorBlock::load(DATA_PATH).unwrap();
        check_block(block.as_ref());
    }

    #[test]
    fn load_buffer() {
        let mut file = std::fs::File::open(DATA_PATH).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let block = metatensor::io::load_block_buffer(&buffer).unwrap();
        check_block(block.as_ref());

        let block = TensorBlock::load_buffer(&buffer).unwrap();
        check_block(block.as_ref());
    }

    #[test]
    fn save() {
        let mut file = std::fs::File::open(DATA_PATH).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let block = metatensor::io::load_block_buffer(&buffer).unwrap();

        let temdir = tempfile::tempdir().unwrap();
        let mut tempath = temdir.path().to_path_buf();
        tempath.push("block.mts");
        block.save(&tempath).unwrap();

        let mut file = std::fs::File::open(tempath).unwrap();
        let mut saved = Vec::new();
        file.read_to_end(&mut saved).unwrap();

        assert_eq!(buffer.len(), saved.len());
        assert_eq!(buffer, saved);
    }

    #[test]
    fn save_buffer() {
        let mut file = std::fs::File::open(DATA_PATH).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let block = metatensor::io::load_block_buffer(&buffer).unwrap();

        let mut saved = Vec::new();
        metatensor::io::save_block_buffer(block.as_ref(), &mut saved).unwrap();
        assert_eq!(buffer.len(), saved.len());
        assert_eq!(buffer, saved);

        saved.clear();
        block.save_buffer(&mut saved).unwrap();

        assert_eq!(buffer.len(), saved.len());
        assert_eq!(buffer, saved);
    }


    fn check_block(block: TensorBlockRef) {
        assert_eq!(block.values().device().unwrap(), dlpk::sys::DLDevice::cpu());
        assert_eq!(block.values().shape().unwrap(), [9, 5, 3]);
        assert_eq!(block.samples().names(), ["system", "atom"]);
        assert_eq!(block.components().len(), 1);
        assert_eq!(block.components()[0].names(), ["o3_mu"]);
        assert_eq!(block.properties().names(), ["n"]);

        assert_eq!(block.gradient_list(), ["positions"]);
        let gradient = block.gradient("positions").unwrap();
        assert_eq!(gradient.values().device().unwrap(), dlpk::sys::DLDevice::cpu());
        assert_eq!(gradient.values().shape().unwrap(), [59, 3, 5, 3]);
        assert_eq!(gradient.samples().names(), ["sample", "system", "atom"]);
        assert_eq!(gradient.components().len(), 2);
        assert_eq!(gradient.components()[0].names(), ["xyz"]);
        assert_eq!(gradient.components()[1].names(), ["o3_mu"]);
        assert_eq!(gradient.properties().names(), ["n"]);
    }
}

/// Serialization tests for different data types.
///
/// The write path in metatensor-core/src/io/block.rs dispatches on the DLPack
/// dtype of the source array. The read path dispatches on the NPY type
/// descriptor *and* the target array's DLPack dtype. These tests exercise
/// the native-endian write path for every supported scalar type, and full
/// round-trip for f64 (the type produced by the default create_array callback).
mod dtype_serialization {
    use metatensor::{Labels, TensorBlock};

    fn make_labels() -> (Labels, Labels) {
        let samples = Labels::new(["s"], vec![[0], [1]]);
        let properties = Labels::new(["p"], vec![[0], [1], [2]]);
        (samples, properties)
    }

    /// Verify that saving a block with a given dtype succeeds and produces
    /// a non-empty buffer. This exercises the write_data match arms.
    macro_rules! save_dtype_test {
        ($name:ident, $ty:ty, $val:expr) => {
            #[test]
            fn $name() {
                let (samples, properties) = make_labels();
                let data = ndarray::Array::<$ty, _>::from_elem(vec![2, 3], $val);
                let block = TensorBlock::new(data, &samples, &[], &properties).unwrap();

                let mut buf = Vec::new();
                metatensor::io::save_block_buffer(block.as_ref(), &mut buf).unwrap();
                assert!(!buf.is_empty());
            }
        };
    }

    save_dtype_test!(save_f32, f32, 1.5_f32);
    save_dtype_test!(save_f64, f64, 2.5_f64);
    save_dtype_test!(save_i8, i8, 42_i8);
    save_dtype_test!(save_i16, i16, 1000_i16);
    save_dtype_test!(save_i32, i32, 100_000_i32);
    save_dtype_test!(save_i64, i64, 1_000_000_i64);
    save_dtype_test!(save_u8, u8, 200_u8);
    save_dtype_test!(save_u16, u16, 50_000_u16);
    save_dtype_test!(save_u32, u32, 3_000_000_u32);
    save_dtype_test!(save_u64, u64, 9_000_000_u64);
    save_dtype_test!(save_bool, bool, true);
    save_dtype_test!(save_f16, half::f16, half::f16::from_f32(1.0));
    save_dtype_test!(save_complex64, [f32; 2], [1.0_f32, 2.0_f32]);
    save_dtype_test!(save_complex128, [f64; 2], [3.0_f64, 4.0_f64]);

    /// Full round-trip for f64 (the default create_array type).
    #[test]
    fn f64_roundtrip() {
        let (samples, properties) = make_labels();
        let data = ndarray::Array::<f64, _>::from_elem(vec![2, 3], 3.41);
        let block = TensorBlock::new(data, &samples, &[], &properties).unwrap();

        let mut buf = Vec::new();
        metatensor::io::save_block_buffer(block.as_ref(), &mut buf).unwrap();

        let loaded = metatensor::io::load_block_buffer(&buf).unwrap();
        assert_eq!(
            block.values().shape().unwrap(),
            loaded.values().shape().unwrap(),
        );
    }

    /// Verify that an empty (zero-element) array round-trips.
    #[test]
    fn empty_array_roundtrip() {
        let samples = Labels::empty(vec!["s"]);
        let properties = Labels::new(["p"], vec![[0], [1]]);
        let data = ndarray::Array::<f64, _>::from_elem(vec![0, 2], 0.0);
        let block = TensorBlock::new(data, &samples, &[], &properties).unwrap();

        let mut buf = Vec::new();
        metatensor::io::save_block_buffer(block.as_ref(), &mut buf).unwrap();

        let loaded = metatensor::io::load_block_buffer(&buf).unwrap();
        assert_eq!(
            block.values().shape().unwrap(),
            loaded.values().shape().unwrap(),
        );
    }
}

mod labels {
    use std::io::Read;
    use metatensor::Labels;

    const DATA_PATH: &str = "../../metatensor-core/tests/keys.mts";

    #[test]
    fn load_file() {
        let labels = metatensor::io::load_labels(DATA_PATH).unwrap();
        check_labels(&labels);

        let labels = Labels::load(DATA_PATH).unwrap();
        check_labels(&labels);
    }

    #[test]
    fn load_buffer() {
        let mut file = std::fs::File::open(DATA_PATH).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let labels = metatensor::io::load_labels_buffer(&buffer).unwrap();
        check_labels(&labels);

        let labels = Labels::load_buffer(&buffer).unwrap();
        check_labels(&labels);
    }

    #[test]
    fn save() {
        let mut file = std::fs::File::open(DATA_PATH).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let labels = metatensor::io::load_labels_buffer(&buffer).unwrap();

        let temdir = tempfile::tempdir().unwrap();
        let mut tempath = temdir.path().to_path_buf();
        tempath.push("labels.mts");
        labels.save(&tempath).unwrap();

        let mut file = std::fs::File::open(tempath).unwrap();
        let mut saved = Vec::new();
        file.read_to_end(&mut saved).unwrap();

        assert_eq!(buffer.len(), saved.len());
        assert_eq!(buffer, saved);
    }

    #[test]
    fn save_buffer() {
        let mut file = std::fs::File::open(DATA_PATH).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let labels = metatensor::io::load_labels_buffer(&buffer).unwrap();

        let mut saved = Vec::new();
        metatensor::io::save_labels_buffer(&labels, &mut saved).unwrap();
        assert_eq!(buffer.len(), saved.len());
        assert_eq!(buffer, saved);

        saved.clear();
        labels.save_buffer(&mut saved).unwrap();
        assert_eq!(buffer.len(), saved.len());
        assert_eq!(buffer, saved);
    }


    fn check_labels(labels: &Labels) {
        assert_eq!(labels.names(), ["o3_lambda", "o3_sigma", "center_type", "neighbor_type"]);
        assert_eq!(labels.count(), 27);
    }
}

mod custom_array {
    use std::io::Read;

    use std::sync::atomic::{AtomicUsize, Ordering};

    use dlpk::sys::{DLDataTypeCode, DLDataType};
    use metatensor::MtsArray;
    use metatensor::c_api::{MTS_CALLBACK_ERROR, MTS_SUCCESS, mts_array_t, mts_status_t};

    const BLOCK_DATA_PATH: &str = "../../metatensor-core/tests/block.mts";
    const TENSOR_DATA_PATH: &str = "../../metatensor-core/tests/data.mts";

    static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
    static CREATE_ARRAY_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    extern "C" fn create_array(shape: *const usize, shape_count: usize, dtype: DLDataType, array: *mut mts_array_t) -> mts_status_t {
        CALL_COUNT.fetch_add(1, Ordering::SeqCst);

        let shape = unsafe { std::slice::from_raw_parts(shape, shape_count) };

        let new_array: MtsArray = match (dtype.code, dtype.bits) {
            (DLDataTypeCode::kDLFloat, 64) => {
                ndarray::Array::<f64, _>::zeros(shape).into()
            }
            (DLDataTypeCode::kDLFloat, 32) => {
                ndarray::Array::<f32, _>::zeros(shape).into()
            }
            (DLDataTypeCode::kDLInt, 32) => {
                ndarray::Array::<i32, _>::zeros(shape).into()
            }
            (DLDataTypeCode::kDLInt, 64) => {
                ndarray::Array::<i64, _>::zeros(shape).into()
            }
            (DLDataTypeCode::kDLUInt, 8) => {
                ndarray::Array::<u8, _>::zeros(shape).into()
            }
            _ => {
                return MTS_CALLBACK_ERROR;
            }
        };

        unsafe {
            *array = new_array.into_raw();
        }

        MTS_SUCCESS
    }

    #[test]
    fn block_load_file() {
        let _guard = CREATE_ARRAY_MUTEX.lock().unwrap();
        CALL_COUNT.store(0, Ordering::SeqCst);

        let block = metatensor::io::load_block_custom_array(BLOCK_DATA_PATH, Some(create_array)).unwrap();
        assert!(CALL_COUNT.load(Ordering::SeqCst) > 0);
        assert_eq!(block.values().shape().unwrap(), [9, 5, 3]);
    }

    #[test]
    fn block_load_buffer() {
        let _guard = CREATE_ARRAY_MUTEX.lock().unwrap();
        CALL_COUNT.store(0, Ordering::SeqCst);

        let mut file = std::fs::File::open(BLOCK_DATA_PATH).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let block = metatensor::io::load_block_buffer_custom_array(&buffer, Some(create_array)).unwrap();

        assert!(CALL_COUNT.load(Ordering::SeqCst) > 0);
        assert_eq!(block.values().shape().unwrap(), [9, 5, 3]);
    }

    #[test]
    fn tensor_load_file() {
        let _guard = CREATE_ARRAY_MUTEX.lock().unwrap();
        CALL_COUNT.store(0, Ordering::SeqCst);

        let tensor = metatensor::io::load_custom_array(TENSOR_DATA_PATH, Some(create_array)).unwrap();

        assert!(CALL_COUNT.load(Ordering::SeqCst) > 0);
        assert_eq!(tensor.keys().count(), 27);
    }

    #[test]
    fn tensor_load_buffer() {
        let _guard = CREATE_ARRAY_MUTEX.lock().unwrap();
        CALL_COUNT.store(0, Ordering::SeqCst);

        let mut file = std::fs::File::open(TENSOR_DATA_PATH).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let tensor = metatensor::io::load_buffer_custom_array(&buffer, Some(create_array)).unwrap();

        assert!(CALL_COUNT.load(Ordering::SeqCst) > 0);
        assert_eq!(tensor.keys().count(), 27);
    }
}

mod error_paths {
    #[test]
    fn load_invalid_buffer_as_tensor() {
        let data = b"this is not a valid mts file";
        let result = metatensor::io::load_buffer(data);
        assert_eq!(
            result.unwrap_err().message,
            "serialization format error: unable to load a TensorMap from buffer: invalid Zip archive: Could not find central directory end: at '<root>'"
        );
    }

    #[test]
    fn load_invalid_buffer_as_block() {
        let data = b"this is not a valid mts file";
        let result = metatensor::io::load_block_buffer(data);
        assert_eq!(
            result.unwrap_err().message,
            "serialization format error: unable to load a TensorBlock from buffer: invalid Zip archive: Could not find central directory end: at '<root>'"
        );
    }

    #[test]
    fn load_invalid_buffer_as_labels() {
        let data = b"this is not a valid mts file";
        let result = metatensor::io::load_labels_buffer(data);
        assert_eq!(
            result.unwrap_err().message,
            "serialization format error: unable to load Labels from buffer: start does not match magic string"
        );
    }

    #[test]
    fn load_nonexistent_file() {
        let result = metatensor::io::load("/nonexistent/path/to/file.mts");
        assert!(result.unwrap_err().message.starts_with("io error:"));
    }

    #[test]
    fn load_block_nonexistent_file() {
        let result = metatensor::io::load_block("/nonexistent/path/to/file.mts");
        assert!(result.unwrap_err().message.starts_with("io error:"));
    }

    #[test]
    fn load_labels_nonexistent_file() {
        let result = metatensor::io::load_labels("/nonexistent/path/to/file.mts");
        assert!(result.unwrap_err().message.starts_with("io error:"));
    }

    #[test]
    fn save_block_to_buffer_roundtrip() {
        let block = metatensor::io::load_block("../../metatensor-core/tests/block.mts").unwrap();
        let mut buf = Vec::new();
        metatensor::io::save_block_buffer(block.as_ref(), &mut buf).unwrap();
        assert!(!buf.is_empty());

        // re-load from buffer
        let reloaded = metatensor::io::load_block_buffer(&buf).unwrap();
        assert_eq!(
            block.values().shape().unwrap(),
            reloaded.values().shape().unwrap(),
        );
    }

    #[test]
    fn save_tensor_to_buffer_roundtrip() {
        let tensor = metatensor::io::load("../../metatensor-core/tests/data.mts").unwrap();
        let mut buf = Vec::new();
        metatensor::io::save_buffer(&tensor, &mut buf).unwrap();
        assert!(!buf.is_empty());

        let reloaded = metatensor::io::load_buffer(&buf).unwrap();
        assert_eq!(tensor.keys().count(), reloaded.keys().count());
    }

    #[test]
    fn save_labels_to_buffer_roundtrip() {
        let labels = metatensor::io::load_labels("../../metatensor-core/tests/keys.mts").unwrap();
        let mut buf = Vec::new();
        metatensor::io::save_labels_buffer(&labels, &mut buf).unwrap();
        assert!(!buf.is_empty());

        let reloaded = metatensor::io::load_labels_buffer(&buf).unwrap();
        assert_eq!(labels.count(), reloaded.count());
        assert_eq!(labels.names(), reloaded.names());
    }

    #[test]
    fn block_data_as_tensor_fails() {
        // Try loading block data as a TensorMap
        let mut file = std::fs::File::open("../../metatensor-core/tests/block.mts").unwrap();
        let mut buffer = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut buffer).unwrap();
        let result = metatensor::io::load_buffer(&buffer);

        assert_eq!(
            result.unwrap_err().message,
            "serialization format error: unable to load a TensorMap from buffer, use `load_block_buffer` to load TensorBlock: specified file not found in archive: at 'keys.npy'"
        );
    }

    #[test]
    fn tensor_data_as_block_fails() {
        // Try loading tensor data as a TensorBlock
        let mut file = std::fs::File::open("../../metatensor-core/tests/data.mts").unwrap();
        let mut buffer = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut buffer).unwrap();
        let result = metatensor::io::load_block_buffer(&buffer);

        assert_eq!(
            result.unwrap_err().message,
            "serialization format error: unable to load a TensorBlock from buffer, use `load` to load TensorMap: specified file not found in archive: at 'values.npy'"
        );
    }

    #[test]
    fn empty_buffer_fails() {
        let result = metatensor::io::load_buffer(b"");
        assert_eq!(
            result.unwrap_err().message,
            "serialization format error: unable to load TensorMap from empty buffer"
        );

        let result = metatensor::io::load_block_buffer(b"");
        assert_eq!(
            result.unwrap_err().message,
            "serialization format error: unable to load TensorBlock from empty buffer"
        );

        let result = metatensor::io::load_labels_buffer(b"");
        assert_eq!(
            result.unwrap_err().message,
            "serialization format error: unable to load Labels from empty buffer"
        );
    }

    #[test]
    fn save_to_tempfile_and_reload() {
        let tensor = metatensor::io::load("../../metatensor-core/tests/data.mts").unwrap();

        let tmpdir = tempfile::tempdir().unwrap();
        let path = tmpdir.path().join("test.mts");
        tensor.save(&path).unwrap();

        let reloaded = metatensor::io::load(path.to_str().unwrap()).unwrap();
        assert_eq!(tensor.keys().count(), reloaded.keys().count());
    }
}
