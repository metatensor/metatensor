#[test]
fn load_file() {
    let tensor = equistore::io::load("equistore-core/tests/data.npz").unwrap();

    assert_eq!(tensor.keys().names(), ["spherical_harmonics_l", "center_species", "neighbor_species"]);
    assert_eq!(tensor.keys().count(), 27);

    let block = tensor.block_by_id(13);

    assert_eq!(block.values().data.as_array().shape(), [9, 3, 3]);
    assert_eq!(block.values().samples.names(), ["structure", "center"]);
    assert_eq!(block.values().components.len(), 1);
    assert_eq!(block.values().components[0].names(), ["spherical_harmonics_m"]);
    assert_eq!(block.values().properties.names(), ["n"]);

    assert_eq!(block.gradient_list(), ["positions"]);
    let gradient = block.gradient("positions").unwrap();
    assert_eq!(gradient.data.as_array().shape(), [27, 3, 3, 3]);
    assert_eq!(gradient.samples.names(), ["sample", "structure", "atom"]);
    assert_eq!(gradient.components.len(), 2);
    assert_eq!(gradient.components[0].names(), ["gradient_direction"]);
    assert_eq!(gradient.components[1].names(), ["spherical_harmonics_m"]);
    assert_eq!(gradient.properties.names(), ["n"]);
}
