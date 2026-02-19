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

        assert_eq!(block.values().as_array().shape(), [9, 3, 3]);
        assert_eq!(block.samples().names(), ["system", "atom"]);
        assert_eq!(block.components().len(), 1);
        assert_eq!(block.components()[0].names(), ["o3_mu"]);
        assert_eq!(block.properties().names(), ["n"]);

        assert_eq!(block.gradient_list(), ["positions"]);
        let gradient = block.gradient("positions").unwrap();
        assert_eq!(gradient.values().as_array().shape(), [27, 3, 3, 3]);
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
        assert_eq!(block.values().as_array().shape(), [9, 5, 3]);
        assert_eq!(block.samples().names(), ["system", "atom"]);
        assert_eq!(block.components().len(), 1);
        assert_eq!(block.components()[0].names(), ["o3_mu"]);
        assert_eq!(block.properties().names(), ["n"]);

        assert_eq!(block.gradient_list(), ["positions"]);
        let gradient = block.gradient("positions").unwrap();
        assert_eq!(gradient.values().as_array().shape(), [59, 3, 5, 3]);
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
    use metatensor::{Labels, LabelsBuilder, TensorBlock};

    fn make_labels() -> (Labels, Labels) {
        let mut samples = LabelsBuilder::new(vec!["s"]);
        samples.add(&[0]);
        samples.add(&[1]);
        let samples = samples.finish();

        let mut properties = LabelsBuilder::new(vec!["p"]);
        properties.add(&[0]);
        properties.add(&[1]);
        properties.add(&[2]);
        let properties = properties.finish();

        (samples, properties)
    }

    /// Verify that saving a block with a given dtype succeeds and produces
    /// a non-empty buffer. This exercises the write_data match arms.
    macro_rules! save_dtype_test {
        ($name:ident, $ty:ty, $val:expr) => {
            #[test]
            fn $name() {
                let (samples, properties) = make_labels();
                let data = ndarray::ArcArray::<$ty, _>::from_elem(vec![2, 3], $val);
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
        let data = ndarray::ArcArray::<f64, _>::from_elem(vec![2, 3], 3.41);
        let block = TensorBlock::new(data, &samples, &[], &properties).unwrap();

        let mut buf = Vec::new();
        metatensor::io::save_block_buffer(block.as_ref(), &mut buf).unwrap();

        let loaded = metatensor::io::load_block_buffer(&buf).unwrap();
        assert_eq!(
            block.values().as_array().shape(),
            loaded.values().as_array().shape(),
        );
    }

    /// Verify that an empty (zero-element) array round-trips.
    #[test]
    fn empty_array_roundtrip() {
        let samples = LabelsBuilder::new(vec!["s"]).finish();
        let mut properties = LabelsBuilder::new(vec!["p"]);
        properties.add(&[0]);
        properties.add(&[1]);
        let properties = properties.finish();
        let data = ndarray::ArcArray::<f64, _>::from_elem(vec![0, 2], 0.0);
        let block = TensorBlock::new(data, &samples, &[], &properties).unwrap();

        let mut buf = Vec::new();
        metatensor::io::save_block_buffer(block.as_ref(), &mut buf).unwrap();

        let loaded = metatensor::io::load_block_buffer(&buf).unwrap();
        assert_eq!(
            block.values().as_array().shape(),
            loaded.values().as_array().shape(),
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

mod error_paths {
    #[test]
    fn load_invalid_buffer_as_tensor() {
        let data = b"this is not a valid mts file";
        let err = metatensor::io::load_buffer(data);
        assert!(err.is_err());
    }

    #[test]
    fn load_invalid_buffer_as_block() {
        let data = b"this is not a valid mts file";
        let err = metatensor::io::load_block_buffer(data);
        assert!(err.is_err());
    }

    #[test]
    fn load_invalid_buffer_as_labels() {
        let data = b"this is not a valid mts file";
        let err = metatensor::io::load_labels_buffer(data);
        assert!(err.is_err());
    }

    #[test]
    fn load_nonexistent_file() {
        let err = metatensor::io::load("/nonexistent/path/to/file.mts");
        assert!(err.is_err());
    }

    #[test]
    fn load_block_nonexistent_file() {
        let err = metatensor::io::load_block("/nonexistent/path/to/file.mts");
        assert!(err.is_err());
    }

    #[test]
    fn load_labels_nonexistent_file() {
        let err = metatensor::io::load_labels("/nonexistent/path/to/file.mts");
        assert!(err.is_err());
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
            block.values().as_array().shape(),
            reloaded.values().as_array().shape(),
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
        // block data does not have keys.npy, so loading as tensor should fail
        assert!(result.is_err());
    }

    #[test]
    fn tensor_data_as_block_fails() {
        // Try loading tensor data as a TensorBlock
        let mut file = std::fs::File::open("../../metatensor-core/tests/data.mts").unwrap();
        let mut buffer = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut buffer).unwrap();
        let result = metatensor::io::load_block_buffer(&buffer);
        // tensor data does not have values.npy at the root, so loading as block should fail
        assert!(result.is_err());
    }

    #[test]
    fn empty_buffer_fails() {
        let result = metatensor::io::load_buffer(b"");
        assert!(result.is_err());

        let result = metatensor::io::load_block_buffer(b"");
        assert!(result.is_err());

        let result = metatensor::io::load_labels_buffer(b"");
        assert!(result.is_err());
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

mod device {
    use metatensor::{LabelsBuilder, TensorBlock};
    use metatensor::Array;
    use dlpk::sys::{DLDevice, DLDeviceType, DLPackVersion};

    #[test]
    fn arcarray_device_is_cpu() {
        let data = ndarray::ArcArray::<f64, _>::zeros(vec![2, 3]);
        assert_eq!(data.device(), DLDevice::cpu());
    }

    #[test]
    fn block_values_device() {
        let mut samples = LabelsBuilder::new(vec!["s"]);
        samples.add(&[0]);
        let samples = samples.finish();

        let mut properties = LabelsBuilder::new(vec!["p"]);
        properties.add(&[0]);
        properties.add(&[1]);
        let properties = properties.finish();

        let data = ndarray::ArcArray::<f64, _>::zeros(vec![1, 2]);
        let block = TensorBlock::new(data, &samples, &[], &properties).unwrap();
        assert_eq!(block.values().as_array().device(), DLDevice::cpu());
    }

    #[test]
    fn loaded_block_device() {
        let block = metatensor::io::load_block("../../metatensor-core/tests/block.mts").unwrap();
        assert_eq!(block.values().as_array().device(), DLDevice::cpu());
    }

    #[test]
    fn as_dlpack_rejects_stream() {
        let data = ndarray::ArcArray::<f64, _>::zeros(vec![2, 3]);
        match data.as_dlpack(DLDevice::cpu(), Some(42), DLPackVersion::current()) {
            Err(e) => assert!(e.message.contains("stream"), "{}", e.message),
            Ok(_) => panic!("expected error for non-null stream"),
        }
    }

    #[test]
    fn as_dlpack_rejects_wrong_device() {
        let data = ndarray::ArcArray::<f64, _>::zeros(vec![2, 3]);
        let cuda = DLDevice {
            device_type: DLDeviceType::kDLCUDA,
            device_id: 0,
        };
        match data.as_dlpack(cuda, None, DLPackVersion::current()) {
            Err(e) => assert!(e.message.contains("does not match"), "{}", e.message),
            Ok(_) => panic!("expected error for CUDA device on CPU array"),
        }
    }

    #[test]
    fn as_dlpack_rejects_incompatible_version() {
        let data = ndarray::ArcArray::<f64, _>::zeros(vec![2, 3]);
        let bad_version = DLPackVersion { major: 99, minor: 0 };
        match data.as_dlpack(DLDevice::cpu(), None, bad_version) {
            Err(e) => assert!(e.message.contains("version"), "{}", e.message),
            Ok(_) => panic!("expected error for incompatible DLPack version"),
        }
    }

    #[test]
    fn as_dlpack_success() {
        let data = ndarray::ArcArray::<f64, _>::zeros(vec![2, 3]);
        let tensor = data
            .as_dlpack(DLDevice::cpu(), None, DLPackVersion::current())
            .expect("as_dlpack should succeed for CPU f64 array");
        let tensor_ref = tensor.as_ref();
        assert_eq!(tensor_ref.raw.dtype.bits, 64);
    }
}
