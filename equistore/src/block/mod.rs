// TensorBlock can be manipulated in three forms: as standalone block owning
// it's data, or as a reference (or mutable reference) to a block inside a
// `TensorMap`. The three forms are represented by a pointer to `eqs_block_t`
// in the C API, but we wrap them into three different types for Rust.

mod block_ref;
pub use self::block_ref::{TensorBlockRef, BasicBlock, GradientsIter};

mod block_mut;
pub use self::block_mut::{TensorBlockRefMut, BasicBlockMut, GradientsMutIter};

mod owned;
pub use self::owned::TensorBlock;
