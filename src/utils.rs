use std::ffi::{CString, CStr};


/// An analog to `std::ffi::CString` that is immutable & can be shared between
/// traits safely. This is used to store the columns names in a set of `Labels`
/// in a C-compatible way.
#[repr(transparent)]
pub struct ConstCString(*const std::os::raw::c_char);

impl ConstCString {
    /// Create a new `ConstCString` containing the same data as the given `str`.
    pub fn new(str: CString) -> ConstCString {
        ConstCString(CString::into_raw(str))
    }

    /// Get the content of this `ConstCString` as a `Cstr` reference
    pub fn as_c_str(&self) -> &CStr {
        // SAFETY: `CStr::from_ptr` is OK since we created this pointer with
        // `CString::into_raw`, which fulfils all the requirements of
        // `CStr::from_ptr`
        unsafe {
            CStr::from_ptr(self.0)
        }
    }

    /// Get the content of this `ConstCString` as a `str` reference, panicking
    /// if this `ConstCString` contains invalid UTF8.
    pub fn as_str(&self) -> &str {
        return self.as_c_str().to_str().expect("invalid UTF8");
    }
}

impl Drop for ConstCString {
    fn drop(&mut self) {
        // SAFETY: `CString::from_raw` is OK since we created this pointer with
        // `CString::into_raw`
        unsafe {
            let str = CString::from_raw(self.0 as *mut _);
            drop(str);
        }
    }
}

impl PartialEq for ConstCString {
    fn eq(&self, other: &Self) -> bool {
        self.as_c_str() == other.as_c_str()
    }
}

impl std::cmp::Eq for ConstCString {}

impl Clone for ConstCString {
    fn clone(&self) -> Self {
        let str = self.as_c_str().to_owned();
        return ConstCString::new(str);
    }
}

// SAFETY: `ConstCString` is immutable, so sharing between threads causes no
// issue
unsafe impl Sync for ConstCString {}
unsafe impl Send for ConstCString {}

impl std::fmt::Debug for ConstCString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ConstCString").field(&self.as_c_str()).finish()
    }
}
