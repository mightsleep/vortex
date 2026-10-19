// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "benchmark fixtures use indices that fit in i32"
)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_mask::Mask;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

// Smaller than primitive_zip: the per-run generic path costs ~150 ns per row on an alternating mask,
// and CodSpeed's simulation runs ~10x the local walltime.
const LEN: usize = 4_096;

/// Alternating mask: every row is its own run, the worst case for the per-run slice copy.
fn mask() -> Mask {
    Mask::from_iter((0..LEN).map(|i| i.is_multiple_of(2)))
}

/// `CASE WHEN c THEN a ELSE b END` over two decimal columns.
#[divan::bench(types = [i64, i128])]
fn array_vs_array<T: NativeDecimalType + From<i32>>(bencher: Bencher) {
    let dtype = DecimalDType::new(T::MAX_PRECISION, 2);
    run(
        bencher,
        values::<T>(0, dtype),
        values::<T>(1_000_000, dtype),
    );
}

/// `CASE WHEN c THEN a ELSE 0 END`: the literal side is a constant whose canonical form uses the
/// smallest values type for the precision (i64 for precision 15), while the column is i128 as it
/// comes from Arrow's Decimal128.
#[divan::bench]
fn i128_column_vs_literal(bencher: Bencher) {
    let dtype = DecimalDType::new(15, 2);
    let literal = ConstantArray::new(
        Scalar::zero_value(&DType::Decimal(dtype, Nullability::NonNullable)),
        LEN,
    )
    .into_array();
    run(bencher, values::<i128>(0, dtype), literal);
}

fn run(bencher: Bencher, if_true: ArrayRef, if_false: ArrayRef) {
    let mask = mask();
    bencher
        .with_inputs(|| {
            (
                if_true.clone(),
                if_false.clone(),
                mask.clone().into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(t, f, m, ctx)| {
            m.zip(t.clone(), f.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap();
        });
}

fn values<T: NativeDecimalType + From<i32>>(base: i32, dtype: DecimalDType) -> ArrayRef {
    let values: Buffer<T> = (0..LEN as i32)
        .map(|i| <T as From<i32>>::from(base + i))
        .collect();
    DecimalArray::new(values, dtype, Validity::NonNullable).into_array()
}
