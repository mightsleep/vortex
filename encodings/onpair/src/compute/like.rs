// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use onpair::search;
use onpair::search::ContainsTable;
use onpair::search::PrefixQuery;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::like::LikeKernel;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::array::dict_view;
use crate::decode::collect_codes_window;

/// Longest `%needle%` that [`ContainsTable`] accepts.
const MAX_CONTAINS_LEN: usize = 255;

impl LikeKernel for OnPair {
    fn like(
        array: ArrayView<'_, Self>,
        pattern: &ArrayRef,
        options: LikeOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if options.case_insensitive {
            return Ok(None);
        }
        let Some(constant) = pattern.as_constant() else {
            return Ok(None);
        };
        let pattern = match constant.dtype() {
            DType::Utf8(_) => constant
                .as_utf8()
                .value()
                .map(|value| value.as_bytes().to_vec()),
            DType::Binary(_) => constant
                .as_binary()
                .value()
                .map(|value| value.as_slice().to_vec()),
            _ => return Ok(None),
        };
        let Some(pattern) = pattern else {
            return Ok(None);
        };
        let Some(kind) = LikeKind::parse(&pattern) else {
            return Ok(None);
        };
        if let LikeKind::Contains(needle) = kind
            && needle.len() > MAX_CONTAINS_LEN
        {
            return Ok(None);
        }

        let dict = dict_view(array, ctx)?;
        let window = collect_codes_window(array, ctx)?;
        let negated = options.negated;
        let buffer = match kind {
            LikeKind::Eq(needle) => {
                let query = search::tokenize(needle, dict);
                BitBuffer::collect_bool(array.len(), |i| {
                    search::equals(window.row(i), &query) != negated
                })
            }
            LikeKind::StartsWith(prefix) => {
                let query = PrefixQuery::new(prefix, dict);
                BitBuffer::collect_bool(array.len(), |i| {
                    search::starts_with(window.row(i), &query) != negated
                })
            }
            LikeKind::Contains(needle) => {
                let table = ContainsTable::new(needle, dict);
                BitBuffer::collect_bool(array.len(), |i| {
                    search::contains(window.row(i), &table) != negated
                })
            }
        };

        Ok(Some(
            BoolArray::new(
                buffer,
                array
                    .validity()?
                    .union_nullability(constant.dtype().nullability()),
            )
            .into_array(),
        ))
    }
}

/// The LIKE patterns that map onto `onpair::search`, classified as in `vortex-array`'s
/// `LikePattern`. A pattern with an escape, `_`, or `%` anywhere else (`%suffix` included) is
/// left to the canonical path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LikeKind<'a> {
    /// No wildcards: the whole value.
    Eq(&'a [u8]),
    /// `prefix%`
    StartsWith(&'a [u8]),
    /// `%needle%`
    Contains(&'a [u8]),
}

impl<'a> LikeKind<'a> {
    fn parse(pattern: &'a [u8]) -> Option<Self> {
        let plain = |bytes: &[u8]| !bytes.iter().any(|b| matches!(b, b'%' | b'_' | b'\\'));
        if plain(pattern) {
            return Some(Self::Eq(pattern));
        }
        let head = pattern.strip_suffix(b"%")?;
        if plain(head) {
            return Some(Self::StartsWith(head));
        }
        let needle = head.strip_prefix(b"%")?;
        plain(needle).then_some(Self::Contains(needle))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar_fn::fns::like::Like;
    use vortex_array::scalar_fn::fns::like::LikeKernel;
    use vortex_array::scalar_fn::fns::like::LikeOptions;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::LikeKind;
    use crate::DEFAULT_CONFIG;
    use crate::OnPair;
    use crate::OnPairArray;
    use crate::compress::onpair_compress;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    fn onpair(strings: &[Option<&str>], nullability: Nullability) -> VortexResult<OnPairArray> {
        let input = VarBinArray::from_iter(strings.iter().copied(), DType::Utf8(nullability));
        let mut ctx = SESSION.create_execution_ctx();
        onpair_compress(input.as_array(), DEFAULT_CONFIG, &mut ctx)?
            .try_downcast::<OnPair>()
            .map_err(|array| vortex_err!("expected OnPair array, got {}", array.encoding_id()))
    }

    /// Runs the kernel directly, so a silent fallback to the canonical path fails the test.
    fn kernel_like(
        array: &OnPairArray,
        pattern: &str,
        options: LikeOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        let mut ctx = SESSION.create_execution_ctx();
        let pattern = ConstantArray::new(pattern, array.len()).into_array();
        <OnPair as LikeKernel>::like(array.as_view(), &pattern, options, &mut ctx)
    }

    #[test]
    fn parse_like_kinds() {
        assert_eq!(LikeKind::parse(b"abc"), Some(LikeKind::Eq(b"abc")));
        assert_eq!(LikeKind::parse(b""), Some(LikeKind::Eq(b"")));
        assert_eq!(
            LikeKind::parse(b"http%"),
            Some(LikeKind::StartsWith(b"http"))
        );
        assert_eq!(LikeKind::parse(b"%"), Some(LikeKind::StartsWith(b"")));
        assert_eq!(
            LikeKind::parse(b"%needle%"),
            Some(LikeKind::Contains(b"needle"))
        );
        assert_eq!(LikeKind::parse(b"%%"), Some(LikeKind::Contains(b"")));
        assert_eq!(LikeKind::parse(b"%suffix"), None);
        assert_eq!(LikeKind::parse(b"a_c"), None);
        assert_eq!(LikeKind::parse(b"a%c"), None);
        assert_eq!(LikeKind::parse(b"%a%c%"), None);
        assert_eq!(LikeKind::parse(br"\%%"), None);
        assert_eq!(LikeKind::parse(br"%a\_b%"), None);
    }

    /// Every supported shape, including needles that start, end or cross token boundaries, gives
    /// the same answer as the canonical LIKE over the decoded strings.
    #[cfg_attr(miri, ignore)]
    #[rstest]
    #[case("http://example.com/a")]
    #[case("http%")]
    #[case("https://%")]
    #[case("%example%")]
    #[case("%ample.c%")]
    #[case("%/%")]
    #[case("%m/b?x=1%")]
    #[case("%zzz%")]
    #[case("%")]
    #[case("%%")]
    #[case("")]
    fn like_matches_canonical(#[case] pattern: &str) -> VortexResult<()> {
        let strings = [
            Some("http://example.com/a"),
            Some("https://example.com/b?x=1"),
            None,
            Some("ftp://files.example.org"),
            Some(""),
            Some("http://test.org/example"),
            Some("example"),
            Some("https://vortex.dev/m/b?x=1"),
        ];
        let array = onpair(&strings, Nullability::Nullable)?;
        let decoded =
            VarBinArray::from_iter(strings.iter().copied(), DType::Utf8(Nullability::Nullable))
                .into_array();
        let mut ctx = SESSION.create_execution_ctx();
        for negated in [false, true] {
            let options = LikeOptions {
                negated,
                case_insensitive: false,
            };
            let actual = kernel_like(&array, pattern, options)?
                .ok_or_else(|| vortex_err!("kernel declined {pattern:?}"))?;
            let pattern = ConstantArray::new(pattern, decoded.len()).into_array();
            let expected = Like::try_new(decoded.clone(), pattern, options)?.into_array();
            assert_arrays_eq!(actual, expected, &mut ctx);
        }
        Ok(())
    }

    #[cfg_attr(miri, ignore)]
    #[rstest]
    #[case("%suffix")]
    #[case("a_c")]
    #[case(r"http\%%")]
    fn like_declines_other_patterns(#[case] pattern: &str) -> VortexResult<()> {
        let array = onpair(&[Some("a_c"), Some("http%x")], Nullability::NonNullable)?;
        assert!(kernel_like(&array, pattern, LikeOptions::default())?.is_none());
        Ok(())
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn ilike_declines() -> VortexResult<()> {
        let array = onpair(&[Some("Hello")], Nullability::NonNullable)?;
        let options = LikeOptions {
            negated: false,
            case_insensitive: true,
        };
        assert!(kernel_like(&array, "hello%", options)?.is_none());
        Ok(())
    }

    /// `ContainsTable` takes at most 255 bytes; a longer needle goes to the canonical path and
    /// still matches through the `Like` expression.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn long_contains_needle_falls_back() -> VortexResult<()> {
        let long = "x".repeat(300);
        let row = format!("a{long}b");
        let array = onpair(&[Some(&row), Some("ab")], Nullability::NonNullable)?;
        let pattern = format!("%{long}%");
        assert!(kernel_like(&array, &pattern, LikeOptions::default())?.is_none());

        let mut ctx = SESSION.create_execution_ctx();
        let len = array.len();
        let result = Like::try_new(
            array.into_array(),
            ConstantArray::new(pattern.as_str(), len).into_array(),
            LikeOptions::default(),
        )?
        .into_array();
        assert_arrays_eq!(result, BoolArray::from_iter([true, false]), &mut ctx);
        Ok(())
    }

    /// Row windows are relative to the first code offset, which is nonzero for a slice.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn like_on_sliced_array() -> VortexResult<()> {
        let array = onpair(
            &[
                Some("http://a.org"),
                Some("ftp://b.org"),
                Some("http://c.org"),
                Some("http://d.net"),
            ],
            Nullability::NonNullable,
        )?;
        let sliced = array
            .into_array()
            .slice(1..4)?
            .try_downcast::<OnPair>()
            .map_err(|_| vortex_err!("slice dropped OnPair encoding"))?;
        let mut ctx = SESSION.create_execution_ctx();

        let prefix = kernel_like(&sliced, "http%", LikeOptions::default())?
            .ok_or_else(|| vortex_err!("kernel declined prefix"))?;
        assert_arrays_eq!(prefix, BoolArray::from_iter([false, true, true]), &mut ctx);

        let contains = kernel_like(&sliced, "%.org%", LikeOptions::default())?
            .ok_or_else(|| vortex_err!("kernel declined contains"))?;
        assert_arrays_eq!(
            contains,
            BoolArray::from_iter([true, true, false]),
            &mut ctx
        );
        Ok(())
    }
}
