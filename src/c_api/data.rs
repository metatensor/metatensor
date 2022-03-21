use std::os::raw::c_char;
use std::ffi::CStr;

use crate::aml_data_origin_t;

use super::{aml_status_t, catch_unwind};
use super::utils::copy_str_to_c;


/// Register a new data origin with the given `name`. Calling this function
/// multiple times with the same name will give the same `aml_data_origin_t`.
///
/// @param name name of the data origin as an UTF-8 encoded NULL-terminated string
/// @param origin pointer to an `aml_data_origin_t` where the origin will be stored
///
/// @returns The status code of this operation. If the status is not
///          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn aml_register_data_origin(
    name: *const c_char,
    origin: *mut aml_data_origin_t,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(name, origin);

        let name = CStr::from_ptr(name).to_str().unwrap();
        *origin = crate::register_data_origin(name.into());

        Ok(())
    })
}


#[allow(clippy::doc_markdown)]
/// Get the name used to register a given data `origin` in the given `buffer`
///
/// @param origin pre-registered data origin
/// @param buffer buffer to be filled with the data origin name. The origin name
///               will be written  as an UTF-8 encoded, NULL-terminated string
/// @param buffer_size size of the buffer
///
/// @returns The status code of this operation. If the status is not
///          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
///          error message.
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
