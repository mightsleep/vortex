// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::BigCast;
use crate::dtype::NativeDecimalType;
use crate::match_each_decimal_value_type;
use crate::scalar_fn::fns::zip::ZipKernel;
use crate::scalar_fn::fns::zip::select_values;
use crate::scalar_fn::fns::zip::zip_validity;

/// Selects values branchlessly per row, like the primitive zip kernel.
///
/// Without it decimals take the generic zip path, which appends one slice per mask run and costs
/// on the order of 150 ns per row on fragmented masks.
impl ZipKernel for Decimal {
    fn zip(
        if_true: ArrayView<'_, Decimal>,
        if_false: &ArrayRef,
        mask: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(if_false) = if_false.as_opt::<Decimal>() else {
            return Ok(None);
        };

        // Null mask entries select `if_false`, matching `Zip`'s SQL ELSE semantics.
        let mask = mask.clone().fill_null(false)?.execute::<Mask>(ctx)?;
        if !matches!(mask, Mask::Values(_)) {
            return Ok(None);
        }

        let validity = zip_validity(if_true.validity()?, if_false.validity()?, &mask)?;

        // A literal side canonicalizes to the smallest values type for its precision, while a
        // column read through Arrow is usually i128, so widen the narrower side.
        let values_type = if_true.values_type().max(if_false.values_type());
        let array = match_each_decimal_value_type!(values_type, |T| {
            let true_values = widened::<T>(if_true);
            let false_values = widened::<T>(if_false);
            let values = select_values::<T>(&true_values, &false_values, &mask);
            DecimalArray::new(values.freeze(), if_true.decimal_dtype(), validity).into_array()
        });
        Ok(Some(array))
    }
}

/// Values of `array` as `T`, which must be at least as wide as the array's values type.
fn widened<T: NativeDecimalType>(array: ArrayView<'_, Decimal>) -> Buffer<T> {
    if array.values_type() == T::DECIMAL_TYPE {
        return array.buffer::<T>();
    }
    match_each_decimal_value_type!(array.values_type(), |F| {
        array
            .buffer::<F>()
            .iter()
            .map(|&value| {
                <T as BigCast>::from(value)
                    .vortex_expect("widening a decimal value cannot overflow")
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "test fixtures use small indices"
    )]

    use rstest::rstest;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::ArrayRef;
    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::RecursiveCanonical;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::ConstantArray;
    use crate::arrays::Decimal;
    use crate::arrays::DecimalArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::BigCast;
    use crate::dtype::DecimalDType;
    use crate::dtype::NativeDecimalType;
    use crate::dtype::Nullability;
    use crate::dtype::i256;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;

    const LEN: usize = 150;

    fn bits() -> Vec<bool> {
        (0..LEN).map(|i| i.is_multiple_of(3) || i == 64).collect()
    }

    fn zip(
        if_true: ArrayRef,
        if_false: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let result = Mask::from_iter(bits())
            .into_array()
            .zip(if_true, if_false)?
            .execute::<ArrayRef>(ctx)?;
        assert!(result.is::<Decimal>());
        Ok(result)
    }

    fn value<T: NativeDecimalType>(v: i64) -> T {
        <T as BigCast>::from(v).vortex_expect("test value fits every values type")
    }

    /// Selects across the chunk boundary (63/64) and the trailing remainder.
    fn selects_per_row<T: NativeDecimalType>(dtype: DecimalDType) -> VortexResult<()> {
        let array = |base: i64| {
            DecimalArray::from_iter((0..LEN as i64).map(|i| value::<T>(base + i)), dtype)
        };
        let mut ctx = array_session().create_execution_ctx();
        let result = zip(array(0).into_array(), array(1_000).into_array(), &mut ctx)?;

        let bits = bits();
        let expected = DecimalArray::from_iter(
            (0..LEN).map(|i| value::<T>(if bits[i] { i as i64 } else { 1_000 + i as i64 })),
            dtype,
        );
        assert_arrays_eq!(result, expected.into_array(), &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::i32(selects_per_row::<i32> as fn(DecimalDType) -> VortexResult<()>, DecimalDType::new(9, 2))]
    #[case::i64(selects_per_row::<i64>, DecimalDType::new(18, 2))]
    #[case::i128(selects_per_row::<i128>, DecimalDType::new(38, 2))]
    #[case::i256(selects_per_row::<i256>, DecimalDType::new(76, 2))]
    fn zip_selects_per_row(
        #[case] check: fn(DecimalDType) -> VortexResult<()>,
        #[case] dtype: DecimalDType,
    ) -> VortexResult<()> {
        check(dtype)
    }

    /// Validity comes from the selected side.
    #[test]
    fn zip_selects_validity() -> VortexResult<()> {
        let dtype = DecimalDType::new(18, 2);
        let if_true = DecimalArray::from_option_iter(
            (0..LEN as i64).map(|i| (i % 4 != 0).then_some(i)),
            dtype,
        );
        let if_false = DecimalArray::from_option_iter(
            (0..LEN as i64).map(|i| (i % 5 != 0).then_some(1_000 + i)),
            dtype,
        );
        let mut ctx = array_session().create_execution_ctx();
        let result = zip(if_true.into_array(), if_false.into_array(), &mut ctx)?;

        let bits = bits();
        let expected = DecimalArray::from_option_iter(
            (0..LEN as i64).map(|i| {
                if bits[i as usize] {
                    (i % 4 != 0).then_some(i)
                } else {
                    (i % 5 != 0).then_some(1_000 + i)
                }
            }),
            dtype,
        );
        assert_arrays_eq!(result, expected.into_array(), &mut ctx);
        Ok(())
    }

    /// Sides with different values types are widened to the wider one.
    #[test]
    fn zip_widens_narrower_side() -> VortexResult<()> {
        let dtype = DecimalDType::new(15, 2);
        let narrow = DecimalArray::from_iter((0..LEN as i64).map(|i| -i), dtype);
        let wide = DecimalArray::from_iter((0..LEN as i128).map(|i| 1_000 + i), dtype);
        let mut ctx = array_session().create_execution_ctx();
        let result = zip(narrow.into_array(), wide.into_array(), &mut ctx)?;

        let bits = bits();
        let expected = DecimalArray::from_iter(
            (0..LEN).map(|i| {
                if bits[i] {
                    -(i as i128)
                } else {
                    1_000 + i as i128
                }
            }),
            dtype,
        );
        assert_arrays_eq!(result, expected.into_array(), &mut ctx);
        Ok(())
    }

    /// An i128 column zipped with a literal, whose canonical form is i64 for precision 15.
    #[test]
    fn zip_widens_literal_side() -> VortexResult<()> {
        let dtype = DecimalDType::new(15, 2);
        let column = DecimalArray::from_iter(0..LEN as i128, dtype);
        let literal = ConstantArray::new(
            Scalar::decimal(DecimalValue::I64(-7), dtype, Nullability::NonNullable),
            LEN,
        );
        let mut ctx = array_session().create_execution_ctx();
        // The first execution step only canonicalizes the literal, so run to the end.
        let result = Mask::from_iter(bits())
            .into_array()
            .zip(column.into_array(), literal.into_array())?
            .execute::<RecursiveCanonical>(&mut ctx)?
            .0
            .into_array();

        let bits = bits();
        let expected = DecimalArray::from_iter(
            (0..LEN).map(|i| if bits[i] { i as i128 } else { -7 }),
            dtype,
        );
        assert_arrays_eq!(result, expected.into_array(), &mut ctx);
        Ok(())
    }
}
