use lambdaworks_math::{field::element::FieldElement, unsigned_integer::element::UnsignedInteger};
use num_traits::Zero;
use proptest::prelude::*;

use crate::{Felt252, FIELD_HIGH, FIELD_LOW};

/// Returns a [`Strategy`] that generates any valid Felt252
fn any_felt252() -> impl Strategy<Value = Felt252> {
    (0..=FIELD_HIGH)
        // turn range into `impl Strategy`
        .prop_map(|x| x)
        // choose second 128-bit limb capped by first one
        .prop_flat_map(|high| {
            let low = if high == FIELD_HIGH {
                (0..FIELD_LOW).prop_map(|x| x).sboxed()
            } else {
                any::<u128>().sboxed()
            };
            (Just(high), low)
        })
        // turn (u128, u128) into limbs array and then into Felt252
        .prop_map(|(high, low)| {
            let limbs = [
                (high >> 64) as u64,
                (high & ((1 << 64) - 1)) as u64,
                (low >> 64) as u64,
                (low & ((1 << 64) - 1)) as u64,
            ];
            FieldElement::new(UnsignedInteger::from_limbs(limbs))
        })
        .prop_map(|value| Felt252 { value })
}

/// Returns a [`Strategy`] that generates any nonzero Felt252
pub fn nonzero_felt252() -> impl Strategy<Value = Felt252> {
    any_felt252().prop_filter("is zero", |x| !x.is_zero())
}

impl Arbitrary for Felt252 {
    type Parameters = ();

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        any_felt252().sboxed()
    }

    type Strategy = SBoxedStrategy<Self>;
}
