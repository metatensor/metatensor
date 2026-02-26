// Benchmark: dtype vtable slot (direct) vs DLPack fallback
//
// Measures the cost difference between:
// 1. Direct dtype vtable callback (single function pointer call)
// 2. DLPack fallback (full as_dlpack export, then read dtype from DLTensor)
//
// Run with: cargo test -p metatensor --test dtype_bench -- --nocapture --ignored

use std::hint::black_box;
use std::time::Instant;

use dlpk::sys::{DLDataType, DLDevice, DLDeviceType, DLManagedTensorVersioned, DLPackVersion};
use metatensor::Array;
use metatensor_sys::{mts_array_t, MTS_SUCCESS};

const ITERS: u64 = 100_000;

/// Call the dtype vtable slot directly
unsafe fn query_dtype_direct(array: &mts_array_t) -> DLDataType {
    let func = array.dtype.expect("dtype slot is NULL");
    let mut dtype = std::mem::zeroed::<DLDataType>();
    let status = func(array.ptr, &mut dtype);
    assert!(status == MTS_SUCCESS);
    dtype
}

/// Query dtype via DLPack export (the fallback path)
unsafe fn query_dtype_via_dlpack(array: &mts_array_t) -> DLDataType {
    // First get device
    let device_func = array.device.expect("device slot is NULL");
    let mut device = DLDevice { device_type: DLDeviceType::kDLCPU, device_id: 0 };
    let status = device_func(array.ptr, &mut device);
    assert!(status == MTS_SUCCESS);

    // Then export via DLPack
    let dlpack_func = array.as_dlpack.expect("as_dlpack is NULL");
    let mut managed: *mut DLManagedTensorVersioned = std::ptr::null_mut();
    let status = dlpack_func(
        array.ptr,
        &mut managed,
        device,
        std::ptr::null(), // no stream
        DLPackVersion::current(),
    );
    assert!(status == MTS_SUCCESS);
    assert!(!managed.is_null());

    let dtype = (*managed).dl_tensor.dtype;

    // Clean up the exported tensor
    if let Some(deleter) = (*managed).deleter {
        deleter(managed);
    }

    dtype
}

#[test]
#[ignore] // run explicitly with --ignored
fn dtype_vtable_vs_dlpack_fallback() {
    // Create an ndarray-backed mts_array_t (has both dtype slot and as_dlpack)
    let data = ndarray::ArcArray::<f64, _>::zeros(vec![10, 20, 30]);
    let array = mts_array_t::from(Box::new(data) as Box<dyn Array>);

    // Verify both paths are available
    assert!(array.dtype.is_some(), "dtype vtable slot should be set");
    assert!(array.as_dlpack.is_some(), "as_dlpack should be set");

    // Warm up
    unsafe {
        for _ in 0..1000 {
            black_box(query_dtype_direct(&array));
            black_box(query_dtype_via_dlpack(&array));
        }
    }

    // Benchmark: direct vtable slot
    let start = Instant::now();
    unsafe {
        for _ in 0..ITERS {
            black_box(query_dtype_direct(&array));
        }
    }
    let elapsed_direct = start.elapsed();

    // Benchmark: DLPack fallback
    let start = Instant::now();
    unsafe {
        for _ in 0..ITERS {
            black_box(query_dtype_via_dlpack(&array));
        }
    }
    let elapsed_fallback = start.elapsed();

    let ns_direct = elapsed_direct.as_nanos() as f64 / ITERS as f64;
    let ns_fallback = elapsed_fallback.as_nanos() as f64 / ITERS as f64;
    let speedup = ns_fallback / ns_direct;

    println!();
    println!("dtype query benchmark ({ITERS} iterations, shape [10, 20, 30])");
    println!("  vtable slot (direct):   {ns_direct:.1} ns/call");
    println!("  DLPack fallback:        {ns_fallback:.1} ns/call");
    println!("  speedup:                {speedup:.1}x");
    println!();

    // Sanity: both paths return the same dtype
    unsafe {
        let d1 = query_dtype_direct(&array);
        let d2 = query_dtype_via_dlpack(&array);
        assert_eq!(d1, d2, "dtype from both paths must match");
    }

    // The direct path should be meaningfully faster (DLPack allocates)
    assert!(
        speedup > 2.0,
        "expected at least 2x speedup from vtable slot, got {speedup:.1}x"
    );
}
