use std::io::Read;

use metatensor::TensorMap;

#[test]
fn load_file() {
    let tensor = metatensor::io::load("./tests/data.npz").unwrap();
    check_tensor(&tensor);

    let tensor = TensorMap::load("./tests/data.npz").unwrap();
    check_tensor(&tensor);
}


#[test]
fn load_buffer() {
    let mut file = std::fs::File::open("./tests/data.npz").unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();

    let tensor = metatensor::io::load_buffer(&buffer).unwrap();
    check_tensor(&tensor);

    let tensor = TensorMap::load_buffer(&buffer).unwrap();
    check_tensor(&tensor);
}

#[test]
fn save_buffer() {
    let mut file = std::fs::File::open("./tests/data.npz").unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();

    let tensor = metatensor::io::load_buffer(&buffer).unwrap();

    let mut saved = Vec::new();
    metatensor::io::save_buffer(&tensor, &mut saved).unwrap();
    assert_eq!(buffer, saved);

    saved.clear();
    tensor.save_buffer(&mut saved).unwrap();
    assert_eq!(buffer, saved);
}


fn check_tensor(tensor: &TensorMap) {
    assert_eq!(tensor.keys().names(), ["spherical_harmonics_l", "center_species", "neighbor_species"]);
    assert_eq!(tensor.keys().count(), 27);

    let block = tensor.block_by_id(13);

    assert_eq!(block.values().as_array().shape(), [9, 3, 3]);
    assert_eq!(block.samples().names(), ["structure", "center"]);
    assert_eq!(block.components().len(), 1);
    assert_eq!(block.components()[0].names(), ["spherical_harmonics_m"]);
    assert_eq!(block.properties().names(), ["n"]);

    assert_eq!(block.gradient_list(), ["positions"]);
    let gradient = block.gradient("positions").unwrap();
    assert_eq!(gradient.values().as_array().shape(), [27, 3, 3, 3]);
    assert_eq!(gradient.samples().names(), ["sample", "structure", "atom"]);
    assert_eq!(gradient.components().len(), 2);
    assert_eq!(gradient.components()[0].names(), ["gradient_direction"]);
    assert_eq!(gradient.components()[1].names(), ["spherical_harmonics_m"]);
    assert_eq!(gradient.properties().names(), ["n"]);
}
