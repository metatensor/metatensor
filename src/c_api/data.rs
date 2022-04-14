use std::os::raw::c_char;
use std::ffi::CStr;

use crate::eqs_data_origin_t;

use super::{eqs_status_t, catch_unwind};
use super::utils::copy_str_to_c;


/// Register a new data origin with the given `name`. Calling this function
/// multiple times with the same name will give the same `eqs_data_origin_t`.
///
/// @param name name of the data origin as an UTF-8 encoded NULL-terminated string
/// @param origin pointer to an `eqs_data_origin_t` where the origin will be stored
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_register_data_origin(
    name: *const c_char,
    origin: *mut eqs_data_origin_t,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(name, origin);

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
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
#[allow(clippy::cast_possible_truncation)]
pub unsafe extern fn eqs_get_data_origin(
    origin: eqs_data_origin_t,
    buffer: *mut c_char,
    buffer_size: u64,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(buffer);

        let origin = crate::get_data_origin(origin);
        return copy_str_to_c(&origin, buffer, buffer_size as usize);
    })
}
