// Test to regenerate test data files
#[cfg(test)]
mod regen_tests {
    use std::io::Read;

    #[test]
    fn regenerate_data_mts() {
        let mut file = std::fs::File::open("../../metatensor-core/tests/data.mts").unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let tensor = metatensor::io::load_buffer(&buffer).unwrap();

        // Save to regenerate with deterministic ordering
        tensor.save("../../metatensor-core/tests/data.mts").unwrap();
        println!("Regenerated data.mts");
    }

    #[test]
    fn regenerate_block_mts() {
        let mut file = std::fs::File::open("../../metatensor-core/tests/block.mts").unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let block = metatensor::io::load_block_buffer(&buffer).unwrap();

        // Save to regenerate with deterministic ordering
        block.save("../../metatensor-core/tests/block.mts").unwrap();
        println!("Regenerated block.mts");
    }

    #[test]
    fn regenerate_keys_mts() {
        let mut file = std::fs::File::open("../../metatensor-core/tests/keys.mts").unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let labels = metatensor::io::load_labels_buffer(&buffer).unwrap();

        // Save to regenerate with deterministic ordering
        labels.save("../../metatensor-core/tests/keys.mts").unwrap();
        println!("Regenerated keys.mts");
    }
}
