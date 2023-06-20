#![cfg_attr(not(feature = "std"), no_std)]

#[allow(unused_imports)]
#[macro_use]
#[cfg(all(not(feature = "std"), feature = "alloc"))]
pub extern crate alloc;

#[cfg(all(test, not(feature = "lambdaworks-felt")))]
mod arbitrary_bigint_felt;
#[cfg(all(test, feature = "lambdaworks-felt"))]
mod arbitrary_lambdaworks;
#[cfg(not(feature = "lambdaworks-felt"))]
mod bigint_felt;
#[cfg(not(feature = "lambdaworks-felt"))]
mod lib_bigint_felt;
#[cfg(feature = "lambdaworks-felt")]
mod lib_lambdaworks;

use core::fmt;

#[cfg(feature = "lambdaworks-felt")]
pub use lib_lambdaworks::Felt252;

#[cfg(not(feature = "lambdaworks-felt"))]
pub use lib_bigint_felt::Felt252;

pub const PRIME_STR: &str = "0x800000000000011000000000000000000000000000000000000000000000001"; // in decimal, this is equal to 3618502788666131213697322783095070105623107215331596699973092056135872020481
pub const FIELD_HIGH: u128 = (1 << 123) + (17 << 64); // this is equal to 10633823966279327296825105735305134080
pub const FIELD_LOW: u128 = 1;

pub(crate) trait FeltOps {
    fn new<T: Into<FeltBigInt<FIELD_HIGH, FIELD_LOW>>>(value: T) -> Self;

    fn modpow(
        &self,
        exponent: &FeltBigInt<FIELD_HIGH, FIELD_LOW>,
        modulus: &FeltBigInt<FIELD_HIGH, FIELD_LOW>,
    ) -> Self;

    fn iter_u64_digits(&self) -> U64Digits;

    #[cfg(any(feature = "std", feature = "alloc"))]
    fn to_signed_bytes_le(&self) -> Vec<u8>;

    #[cfg(any(feature = "std", feature = "alloc"))]
    fn to_bytes_be(&self) -> Vec<u8>;

    fn parse_bytes(buf: &[u8], radix: u32) -> Option<FeltBigInt<FIELD_HIGH, FIELD_LOW>>;

    fn from_bytes_be(bytes: &[u8]) -> Self;

    #[cfg(any(feature = "std", feature = "alloc"))]
    fn to_str_radix(&self, radix: u32) -> String;

    /// Converts [`Felt252`] into a [`BigInt`] number in the range: `(- FIELD / 2, FIELD / 2)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::cairo_felt::Felt252;
    /// # use num_bigint::BigInt;
    /// # use num_traits::Bounded;
    /// let positive = Felt252::new(5);
    /// assert_eq!(positive.to_signed_felt(), Into::<num_bigint::BigInt>::into(5));
    ///
    /// let negative = Felt252::max_value();
    /// assert_eq!(negative.to_signed_felt(), Into::<num_bigint::BigInt>::into(-1));
    /// ```
    fn to_signed_felt(&self) -> BigInt;

    // Converts [`Felt252`]'s representation directly into a [`BigInt`].
    // Equivalent to doing felt.to_biguint().to_bigint().
    fn to_bigint(&self) -> BigInt;

    /// Converts [`Felt252`] into a [`BigUint`] number.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::cairo_felt::Felt252;
    /// # use num_bigint::BigUint;
    /// # use num_traits::{Num, Bounded};
    /// let positive = Felt252::new(5);
    /// assert_eq!(positive.to_biguint(), Into::<num_bigint::BigUint>::into(5_u32));
    ///
    /// let negative = Felt252::max_value();
    /// assert_eq!(negative.to_biguint(), BigUint::from_str_radix("800000000000011000000000000000000000000000000000000000000000000", 16).unwrap());
    /// ```
    fn to_biguint(&self) -> BigUint;

    fn bits(&self) -> u64;

    fn prime() -> BigUint;
}

#[macro_export]
macro_rules! felt_str {
    ($val: expr) => {
        $crate::Felt252::parse_bytes($val.as_bytes(), 10_u32).expect("Couldn't parse bytes")
    };
    ($val: expr, $opt: expr) => {
        $crate::Felt252::parse_bytes($val.as_bytes(), $opt as u32).expect("Couldn't parse bytes")
    };
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseFeltError;

impl fmt::Display for ParseFeltError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{ParseFeltError:?}")
    }
}
