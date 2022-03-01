use std::os::raw::c_char;
use std::ffi::CStr;

use crate::aml_data_origin_t;

use super::{aml_status_t, catch_unwind};
use super::utils::copy_str_to_c;

#[no_mangle]
pub unsafe extern fn aml_register_data_origin(
    name: *const c_char,
    handle: *mut aml_data_origin_t,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(name, handle);

        let name = CStr::from_ptr(name).to_str().unwrap();
        *handle = crate::register_data_origin(name.into());

        Ok(())
    })
}

#[no_mangle]
#[allow(clippy::cast_possible_truncation)]
pub unsafe extern fn aml_get_data_origin(
    origin: aml_data_origin_t,
    buffer: *mut c_char,
    buffer_size: u64,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(buffer);

        let origin = crate::get_data_origin(origin);
        return copy_str_to_c(&origin, buffer, buffer_size as usize);
    })
}
