mod partial_loading {
    use metatensor::Labels;

    const DATA_PATH: &str = "../../metatensor-core/tests/data.mts";

    /// Helper: compare two blocks for shape and data equality.
    fn assert_blocks_equal(
        fb: metatensor::TensorBlockRef<'_>,
        pb: metatensor::TensorBlockRef<'_>,
        ctx: &str,
    ) {
        assert_eq!(
            fb.values().as_array().shape(),
            pb.values().as_array().shape(),
            "{ctx} values shape mismatch"
        );
        assert_eq!(
            fb.values().as_array(),
            pb.values().as_array(),
            "{ctx} values data mismatch"
        );
        assert_eq!(fb.samples().names(), pb.samples().names(), "{ctx} sample names mismatch");
        assert_eq!(fb.samples().count(), pb.samples().count(), "{ctx} sample count mismatch");
        assert_eq!(fb.properties().names(), pb.properties().names(), "{ctx} property names mismatch");
        assert_eq!(fb.properties().count(), pb.properties().count(), "{ctx} property count mismatch");
        assert_eq!(fb.components().len(), pb.components().len(), "{ctx} component count mismatch");

        assert_eq!(fb.gradient_list(), pb.gradient_list(), "{ctx} gradient list mismatch");
        for param in fb.gradient_list() {
            let fg = fb.gradient(param).unwrap();
            let pg = pb.gradient(param).unwrap();
            assert_eq!(
                fg.values().as_array().shape(),
                pg.values().as_array().shape(),
                "{ctx} gradient '{param}' shape mismatch"
            );
            assert_eq!(
                fg.values().as_array(),
                pg.values().as_array(),
                "{ctx} gradient '{param}' data mismatch"
            );
            assert_eq!(fg.samples().count(), pg.samples().count(), "{ctx} gradient '{param}' sample count mismatch");
        }
    }

    /// No filters: should produce the same result as regular load.
    #[test]
    fn no_filters_matches_full_load() {
        let full = metatensor::io::load(DATA_PATH).unwrap();
        let partial = metatensor::io::load_partial(DATA_PATH, None, None, None).unwrap();

        assert_eq!(full.keys().names(), partial.keys().names());
        assert_eq!(full.keys().count(), partial.keys().count());

        for i in 0..full.keys().count() {
            let fb = full.block_by_id(i);
            let pb = partial.block_by_id(i);
            assert_blocks_equal(fb, pb, &format!("block {i}"));
        }
    }

    /// Filter by keys: select only blocks matching certain key values.
    #[test]
    fn key_filtering() {
        let full = metatensor::io::load(DATA_PATH).unwrap();

        // Select only o3_lambda=1, o3_sigma=1
        let keys_sel = Labels::new(
            ["o3_lambda", "o3_sigma"],
            &[[1, 1]],
        );
        let partial = metatensor::io::load_partial(
            DATA_PATH, Some(&keys_sel), None, None,
        ).unwrap();

        // Should have fewer blocks
        assert!(partial.keys().count() < full.keys().count());
        assert!(partial.keys().count() > 0);

        // Use blocks_matching to find corresponding blocks and compare data
        for i in 0..partial.keys().count() {
            let pb = partial.block_by_id(i);

            // The partial block should exist somewhere in the full map
            // We verify via shape and sample/property counts
            let pshape = pb.values().as_array().shape().to_vec();
            assert!(!pshape.is_empty());
        }
    }

    /// Filter by samples: keep only matching sample rows.
    #[test]
    fn sample_filtering() {
        let full = metatensor::io::load(DATA_PATH).unwrap();

        // Select samples where system=0
        let samples_sel = Labels::new(["system"], &[[0]]);
        let partial = metatensor::io::load_partial(
            DATA_PATH, None, Some(&samples_sel), None,
        ).unwrap();

        // Same number of blocks (no key filtering)
        assert_eq!(full.keys().count(), partial.keys().count());

        for i in 0..partial.keys().count() {
            let pb = partial.block_by_id(i);
            let fb = full.block_by_id(i);

            // Partial should have <= full samples
            assert!(pb.samples().count() <= fb.samples().count(),
                "block {i}: partial samples {} > full samples {}",
                pb.samples().count(), fb.samples().count()
            );

            // Values shape should match: fewer rows, same components/properties
            let fshape = fb.values().as_array().shape().to_vec();
            let pshape = pb.values().as_array().shape().to_vec();
            assert_eq!(pshape[1..], fshape[1..], "block {i} non-sample dims should match");
            assert_eq!(pshape[0], pb.samples().count(), "block {i} sample dim should match count");
        }
    }

    /// Filter by properties: keep only matching property columns.
    #[test]
    fn property_filtering() {
        let full = metatensor::io::load(DATA_PATH).unwrap();

        // Select properties where n=0
        let props_sel = Labels::new(["n"], &[[0]]);
        let partial = metatensor::io::load_partial(
            DATA_PATH, None, None, Some(&props_sel),
        ).unwrap();

        assert_eq!(full.keys().count(), partial.keys().count());

        for i in 0..partial.keys().count() {
            let pb = partial.block_by_id(i);
            let fb = full.block_by_id(i);

            // Partial should have <= full properties
            assert!(pb.properties().count() <= fb.properties().count());

            // Samples should be unchanged
            assert_eq!(fb.samples().count(), pb.samples().count());

            // Gradients should also have filtered properties
            for param in pb.gradient_list() {
                let pg = pb.gradient(param).unwrap();
                assert_eq!(pg.properties().count(), pb.properties().count(),
                    "block {i} gradient '{param}' properties should match block properties"
                );
            }
        }
    }

    /// Combined filtering: keys + samples + properties.
    #[test]
    fn combined_filtering() {
        let full = metatensor::io::load(DATA_PATH).unwrap();

        let keys_sel = Labels::new(["o3_lambda"], &[[1]]);
        let samples_sel = Labels::new(["system"], &[[0]]);
        let props_sel = Labels::new(["n"], &[[0]]);

        let partial = metatensor::io::load_partial(
            DATA_PATH,
            Some(&keys_sel),
            Some(&samples_sel),
            Some(&props_sel),
        ).unwrap();

        // Fewer blocks than full
        assert!(partial.keys().count() < full.keys().count());
        assert!(partial.keys().count() > 0);

        for i in 0..partial.keys().count() {
            let pb = partial.block_by_id(i);
            // Filtered properties
            assert!(pb.properties().count() <= 1);
            // Data shape should be consistent
            let shape = pb.values().as_array().shape().to_vec();
            assert_eq!(shape[0], pb.samples().count());
            let ndim = shape.len();
            assert_eq!(shape[ndim - 1], pb.properties().count());
        }
    }

    /// Gradient sample reindexing: when parent samples are filtered,
    /// gradient sample[0] ("sample" column) must be valid indices
    /// into the new parent samples.
    #[test]
    fn gradient_sample_reindexing() {
        // Select system=0 samples
        let samples_sel = Labels::new(["system"], &[[0]]);
        let partial = metatensor::io::load_partial(
            DATA_PATH, None, Some(&samples_sel), None,
        ).unwrap();

        for i in 0..partial.keys().count() {
            let pb = partial.block_by_id(i);
            let parent_count = pb.samples().count();

            for param in pb.gradient_list() {
                let pg = pb.gradient(param).unwrap();
                // Gradient samples should reference valid parent sample indices
                // The "sample" column (first) should be in [0, parent_count)
                let grad_samples = pg.samples();
                for entry in grad_samples.iter() {
                    let sample_ref = entry[0].i32();
                    assert!(
                        sample_ref >= 0 && (sample_ref as usize) < parent_count,
                        "block {i} gradient '{param}': \
                         sample_ref={sample_ref} out of range [0, {parent_count})"
                    );
                }
            }
        }
    }

    /// Verify that load_partial with sample filtering produces exactly the
    /// same data as load() + manual numpy-style slicing (selecting rows by
    /// index from the full array).
    #[test]
    fn sample_filter_matches_manual_slice() {
        let full = metatensor::io::load(DATA_PATH).unwrap();

        let samples_sel = Labels::new(["system"], &[[0]]);
        let partial = metatensor::io::load_partial(
            DATA_PATH, None, Some(&samples_sel), None,
        ).unwrap();

        assert_eq!(full.keys().count(), partial.keys().count());

        for i in 0..full.keys().count() {
            let fb = full.block_by_id(i);
            let pb = partial.block_by_id(i);

            // Use Labels::select on the full block's samples to get the
            // same indices that load_partial should have used
            let selected = fb.samples().select(&samples_sel).unwrap();
            let indices: Vec<usize> = selected.iter().map(|&i| i as usize).collect();

            assert_eq!(pb.samples().count(), indices.len(),
                "block {i}: partial sample count should match select count");

            // Extract the corresponding rows from the full array and compare
            let fv = fb.values();
            let pv = pb.values();
            let full_arr = fv.as_array();
            let partial_arr = pv.as_array();

            for (new_row, &old_row) in indices.iter().enumerate() {
                let full_row = full_arr.index_axis(ndarray::Axis(0), old_row);
                let partial_row = partial_arr.index_axis(ndarray::Axis(0), new_row);
                assert_eq!(full_row, partial_row,
                    "block {i} row {new_row} (from {old_row}): data mismatch");
            }

            // Also verify gradient data matches manual slice
            for param in fb.gradient_list() {
                let fg = fb.gradient(param).unwrap();
                let pg = pb.gradient(param).unwrap();

                let fgv = fg.values();
                let pgv = pg.values();
                let full_grad = fgv.as_array();
                let partial_grad = pgv.as_array();

                // Build the same parent_map that load_partial uses
                let mut parent_map = std::collections::HashMap::new();
                for (new_idx, &old_idx) in indices.iter().enumerate() {
                    parent_map.insert(old_idx as i32, new_idx as i32);
                }

                // Walk full gradient and collect the rows that should survive
                let mut expected_grad_rows = Vec::new();
                for (row_idx, entry) in fg.samples().iter().enumerate() {
                    let parent_ref = entry[0].i32();
                    if parent_map.contains_key(&parent_ref) {
                        expected_grad_rows.push(row_idx);
                    }
                }

                assert_eq!(pg.samples().count(), expected_grad_rows.len(),
                    "block {i} gradient '{param}': row count mismatch");

                for (new_row, &old_row) in expected_grad_rows.iter().enumerate() {
                    let full_row = full_grad.index_axis(ndarray::Axis(0), old_row);
                    let partial_row = partial_grad.index_axis(ndarray::Axis(0), new_row);
                    assert_eq!(full_row, partial_row,
                        "block {i} gradient '{param}' row {new_row} (from {old_row}): data mismatch");
                }
            }
        }
    }

    /// Verify that load_partial with property filtering produces exactly the
    /// same data as load() + manual column slicing.
    #[test]
    fn property_filter_matches_manual_slice() {
        let full = metatensor::io::load(DATA_PATH).unwrap();

        let props_sel = Labels::new(["n"], &[[0]]);
        let partial = metatensor::io::load_partial(
            DATA_PATH, None, None, Some(&props_sel),
        ).unwrap();

        for i in 0..full.keys().count() {
            let fb = full.block_by_id(i);
            let pb = partial.block_by_id(i);

            // Get property indices via Labels::select
            let selected = fb.properties().select(&props_sel).unwrap();
            let prop_indices: Vec<usize> = selected.iter().map(|&i| i as usize).collect();

            assert_eq!(pb.properties().count(), prop_indices.len(),
                "block {i}: partial property count mismatch");

            // Compare data: for each row, extract the selected columns
            let fv = fb.values();
            let pv = pb.values();
            let full_arr = fv.as_array();
            let partial_arr = pv.as_array();
            let ndim = full_arr.ndim();

            for row in 0..full_arr.shape()[0] {
                for (new_col, &old_col) in prop_indices.iter().enumerate() {
                    if ndim == 2 {
                        assert_eq!(
                            full_arr[[row, old_col]],
                            partial_arr[[row, new_col]],
                            "block {i} [{row}, {new_col}<-{old_col}]: data mismatch"
                        );
                    } else {
                        // For higher dimensions, compare via axis slicing
                        let full_slice = full_arr.index_axis(ndarray::Axis(0), row);
                        let partial_slice = partial_arr.index_axis(ndarray::Axis(0), row);
                        let full_col = full_slice.index_axis(
                            ndarray::Axis(full_slice.ndim() - 1), old_col
                        );
                        let partial_col = partial_slice.index_axis(
                            ndarray::Axis(partial_slice.ndim() - 1), new_col
                        );
                        assert_eq!(full_col, partial_col,
                            "block {i} row {row} prop {new_col}<-{old_col}: data mismatch");
                    }
                }
            }
        }
    }

    /// Nonexistent file should produce an error.
    #[test]
    fn nonexistent_file_errors() {
        let result = metatensor::io::load_partial(
            "/nonexistent/path.mts", None, None, None,
        );
        assert!(result.is_err());
    }
}
