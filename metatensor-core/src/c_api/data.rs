use std::os::raw::c_char;
use std::ffi::CStr;

use crate::mts_data_origin_t;

use super::{mts_status_t, catch_unwind};
use super::utils::copy_str_to_c;


/// Register a new data origin with the given `name`. Calling this function
/// multiple times with the same name will give the same `mts_data_origin_t`.
///
/// @param name name of the data origin as an UTF-8 encoded NULL-terminated string
/// @param origin pointer to an `mts_data_origin_t` where the origin will be stored
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn mts_register_data_origin(
    name: *const c_char,
    origin: *mut mts_data_origin_t,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(name, origin);

        let name = CStr::from_ptr(name).to_str().unwrap();
        *origin = crate::register_data_origin(name.into());

        Ok(())
    })
}


/// Get the name used to register a given data `origin` in the given `buffer`
///
/// @param origin pre-registered data origin
/// @param buffer buffer to be filled with the data origin name. The origin name
///               will be written  as an UTF-8 encoded, NULL-terminated string
/// @param buffer_size size of the buffer
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn mts_get_data_origin(
    origin: mts_data_origin_t,
    buffer: *mut c_char,
    buffer_size: usize,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(buffer);

        let origin = crate::get_data_origin(origin);
        return copy_str_to_c(&origin, buffer, buffer_size);
    })
}
