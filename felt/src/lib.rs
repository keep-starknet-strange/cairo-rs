#![cfg_attr(not(feature = "std"), no_std)]

#[allow(unused_imports)]
#[macro_use]
#[cfg(all(not(feature = "std"), feature = "alloc"))]
pub extern crate alloc;

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

use core::{
    convert::Into,
    fmt,
    iter::Sum,
    ops::{
        Add, AddAssign, BitAnd, BitOr, BitXor, Div, Mul, MulAssign, Neg, Rem, Shl, Shr, ShrAssign,
        Sub, SubAssign,
    },
};

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::{string::String, vec::Vec};
#[cfg(feature = "parity-scale-codec")]
use parity_scale_codec::{Decode, Encode};
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

#[derive(Eq, Hash, PartialEq, PartialOrd, Ord, Clone, Deserialize, Default, Serialize)]
#[cfg_attr(feature = "parity-scale-codec", derive(Encode, Decode))]
pub struct Felt252 {
    value: FeltBigInt<FIELD_HIGH, FIELD_LOW>,
}

macro_rules! from_num {
    ($type:ty) => {
        impl From<$type> for Felt252 {
            fn from(value: $type) -> Self {
                Self {
                    value: value.into(),
                }
            }
        }
    };
}

from_num!(i8);
from_num!(i16);
from_num!(i32);
from_num!(i64);
from_num!(i128);
from_num!(isize);
from_num!(u8);
from_num!(u16);
from_num!(u32);
from_num!(u64);
from_num!(u128);
from_num!(usize);
from_num!(BigInt);
from_num!(&BigInt);
from_num!(BigUint);
from_num!(&BigUint);

impl Felt252 {
    pub fn new<T: Into<Felt252>>(value: T) -> Self {
        value.into()
    }
    pub fn modpow(&self, exponent: &Felt252, modulus: &Felt252) -> Self {
        Self {
            value: self.value.modpow(&exponent.value, &modulus.value),
        }
    }
    pub fn iter_u64_digits(&self) -> U64Digits {
        self.value.iter_u64_digits()
    }

    pub fn to_le_bytes(&self) -> [u8; 32] {
        let mut res = [0u8; 32];
        let mut iter = self.iter_u64_digits();
        let (d0, d1, d2, d3) = (
            iter.next().unwrap_or_default().to_le_bytes(),
            iter.next().unwrap_or_default().to_le_bytes(),
            iter.next().unwrap_or_default().to_le_bytes(),
            iter.next().unwrap_or_default().to_le_bytes(),
        );
        res[..8].copy_from_slice(&d0);
        res[8..16].copy_from_slice(&d1);
        res[16..24].copy_from_slice(&d2);
        res[24..].copy_from_slice(&d3);
        res
    }

    pub fn to_be_bytes(&self) -> [u8; 32] {
        let mut bytes = self.to_le_bytes();
        bytes.reverse();
        bytes
    }

    pub fn to_le_digits(&self) -> [u64; 4] {
        let mut iter = self.iter_u64_digits();
        [
            iter.next().unwrap_or_default(),
            iter.next().unwrap_or_default(),
            iter.next().unwrap_or_default(),
            iter.next().unwrap_or_default(),
        ]
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn to_signed_bytes_le(&self) -> Vec<u8> {
        self.value.to_signed_bytes_le()
    }
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn to_bytes_be(&self) -> Vec<u8> {
        self.value.to_bytes_be()
    }

    pub fn parse_bytes(buf: &[u8], radix: u32) -> Option<Self> {
        Some(Self {
            value: FeltBigInt::parse_bytes(buf, radix)?,
        })
    }
    pub fn from_bytes_be(bytes: &[u8]) -> Self {
        Self {
            value: FeltBigInt::from_bytes_be(bytes),
        }
    }
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn to_str_radix(&self, radix: u32) -> String {
        self.value.to_str_radix(radix)
    }

    pub fn to_signed_felt(&self) -> BigInt {
        #[allow(deprecated)]
        self.value.to_signed_felt()
    }

    pub fn to_bigint(&self) -> BigInt {
        #[allow(deprecated)]
        self.value.to_bigint()
    }

    pub fn to_biguint(&self) -> BigUint {
        #[allow(deprecated)]
        self.value.to_biguint()
    }
    pub fn sqrt(&self) -> Self {
        // Based on Tonelli-Shanks' algorithm for finding square roots
        // and sympy's library implementation of said algorithm.
        if self.is_zero() || self.is_one() {
            return self.clone();
        }

        let max_felt = Felt252::max_value();
        let trailing_prime = Felt252::max_value() >> 192; // 0x800000000000011

        let a = self.pow(&trailing_prime);
        let d = (&Felt252::new(3_i32)).pow(&trailing_prime);
        let mut m = Felt252::zero();
        let mut exponent = Felt252::one() << 191_u32;
        let mut adm;
        for i in 0..192_u32 {
            adm = &a * &(&d).pow(&m);
            adm = (&adm).pow(&exponent);
            exponent >>= 1;
            // if adm ≡ -1 (mod CAIRO_PRIME)
            if adm == max_felt {
                m += Felt252::one() << i;
            }
        }
        let root_1 = self.pow(&((trailing_prime + 1_u32) >> 1)) * (&d).pow(&(m >> 1));
        let root_2 = &max_felt - &root_1 + 1_usize;
        if root_1 < root_2 {
            root_1
        } else {
            root_2
        }
    }

    pub fn bits(&self) -> u64 {
        self.value.bits()
    }

    pub fn prime() -> BigUint {
        FeltBigInt::prime()
    }
}

impl Add for Felt252 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
        }
    }
}

impl<'a> Add for &'a Felt252 {
    type Output = Felt252;
    fn add(self, rhs: Self) -> Self::Output {
        Self::Output {
            value: &self.value + &rhs.value,
        }
    }
}

impl<'a> Add<&'a Felt252> for Felt252 {
    type Output = Self;
    fn add(self, rhs: &Self) -> Self::Output {
        Self::Output {
            value: self.value + &rhs.value,
        }
    }
}

impl Add<u32> for Felt252 {
    type Output = Self;
    fn add(self, rhs: u32) -> Self {
        Self {
            value: self.value + rhs,
        }
    }
}

impl Add<usize> for Felt252 {
    type Output = Self;
    fn add(self, rhs: usize) -> Self {
        Self {
            value: self.value + rhs,
        }
    }
}

impl<'a> Add<usize> for &'a Felt252 {
    type Output = Felt252;
    fn add(self, rhs: usize) -> Self::Output {
        Self::Output {
            value: &self.value + rhs,
        }
    }
}

impl Add<u64> for &Felt252 {
    type Output = Felt252;
    fn add(self, rhs: u64) -> Self::Output {
        Self::Output {
            value: &self.value + rhs,
        }
    }
}

// This is special cased and optimized compared to the obvious implementation
// due to `pc_update` relying on this, which makes it a major bottleneck for
// execution. Testing for this function is extensive, comprised of explicit
// edge and special cases testing and property tests, all comparing to the
// more intuitive `(rhs + self).to_u64()` implementation.
// This particular implementation is much more complex than a slightly more
// intuitive one based on a single match. However, this is 8-62% faster
// depending on the case being bencharked, with an average of 32%, so it's
// worth it.
impl Add<&Felt252> for u64 {
    type Output = Option<u64>;

    fn add(self, rhs: &Felt252) -> Option<u64> {
        const PRIME_DIGITS_LE_HI: (u64, u64, u64) =
            (0x0000000000000000, 0x0000000000000000, 0x0800000000000011);
        const PRIME_MINUS_U64_MAX_DIGITS_LE_HI: (u64, u64, u64) =
            (0xffffffffffffffff, 0xffffffffffffffff, 0x0800000000000010);

        // Iterate through the 64 bits digits in little-endian order to
        // characterize how the sum will behave.
        let mut rhs_digits = rhs.iter_u64_digits();
        // No digits means `rhs` is `0`, so the sum is simply `self`.
        let Some(low) = rhs_digits.next() else {
            return Some(self);
        };
        // A single digit means this is effectively the sum of two `u64` numbers.
        let Some(h0) = rhs_digits.next() else {
            return self.checked_add(low)
        };
        // Now we need to compare the 3 most significant digits.
        // There are two relevant cases from now on, either `rhs` behaves like a
        // substraction of a `u64` or the result of the sum falls out of range.
        let (h1, h2) = (rhs_digits.next()?, rhs_digits.next()?);
        match (h0, h1, h2) {
            // The 3 MSB only match the prime for Felt252::max_value(), which is -1
            // in the signed field, so this is equivalent to substracting 1 to `self`.
            #[allow(clippy::suspicious_arithmetic_impl)]
            PRIME_DIGITS_LE_HI => self.checked_sub(1),
            // For the remaining values between `[-u64::MAX..0]` (where `{0, -1}` have
            // already been covered) the MSB matches that of `PRIME - u64::MAX`.
            // Because we're in the negative number case, we count down. Because `0`
            // and `-1` correspond to different MSBs, `0` and `1` in the LSB are less
            // than `-u64::MAX`, the smallest value we can add to (read, substract it's
            // magnitude from) a `u64` number, meaning we exclude them from the valid
            // case.
            // For the remaining range, we make take the absolute value module-2 while
            // correcting by substracting `1` (note we actually substract `2` because
            // the absolute value itself requires substracting `1`.
            #[allow(clippy::suspicious_arithmetic_impl)]
            PRIME_MINUS_U64_MAX_DIGITS_LE_HI if low >= 2 => {
                (self).checked_sub(u64::MAX - (low - 2))
            }
            // Any other case will result in an addition that is out of bounds, so
            // the addition fails, returning `None`.
            _ => None,
        }
    }
}

impl AddAssign for Felt252 {
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl<'a> AddAssign<&'a Felt252> for Felt252 {
    fn add_assign(&mut self, rhs: &Self) {
        self.value += &rhs.value;
    }
}

impl Sum for Felt252 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Felt252::zero(), |mut acc, x| {
            acc += x;
            acc
        })
    }
}

impl Neg for Felt252 {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            value: self.value.neg(),
        }
    }
}

impl<'a> Neg for &'a Felt252 {
    type Output = Felt252;
    fn neg(self) -> Self::Output {
        Self::Output {
            value: (&self.value).neg(),
        }
    }
}

impl Sub for Felt252 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
        }
    }
}

impl<'a> Sub for &'a Felt252 {
    type Output = Felt252;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output {
            value: &self.value - &rhs.value,
        }
    }
}

impl<'a> Sub<&'a Felt252> for Felt252 {
    type Output = Self;
    fn sub(self, rhs: &Self) -> Self {
        Self {
            value: self.value - &rhs.value,
        }
    }
}

impl Sub<&Felt252> for usize {
    type Output = Felt252;
    fn sub(self, rhs: &Self::Output) -> Self::Output {
        Self::Output {
            value: self - &rhs.value,
        }
    }
}

impl SubAssign for Felt252 {
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value
    }
}

impl<'a> SubAssign<&'a Felt252> for Felt252 {
    fn sub_assign(&mut self, rhs: &Self) {
        self.value -= &rhs.value;
    }
}

impl Sub<u32> for Felt252 {
    type Output = Self;
    fn sub(self, rhs: u32) -> Self {
        Self {
            value: self.value - rhs,
        }
    }
}

impl<'a> Sub<u32> for &'a Felt252 {
    type Output = Felt252;
    fn sub(self, rhs: u32) -> Self::Output {
        Self::Output {
            value: &self.value - rhs,
        }
    }
}

impl Sub<usize> for Felt252 {
    type Output = Self;
    fn sub(self, rhs: usize) -> Self {
        Self {
            value: self.value - rhs,
        }
    }
}

impl Mul for Felt252 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
        }
    }
}

impl<'a> Mul for &'a Felt252 {
    type Output = Felt252;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::Output {
            value: &self.value * &rhs.value,
        }
    }
}

impl<'a> Mul<&'a Felt252> for Felt252 {
    type Output = Self;
    fn mul(self, rhs: &Self) -> Self {
        Self {
            value: self.value * &rhs.value,
        }
    }
}

impl<'a> MulAssign<&'a Felt252> for Felt252 {
    fn mul_assign(&mut self, rhs: &Self) {
        self.value *= &rhs.value;
    }
}

impl Pow<u32> for Felt252 {
    type Output = Self;
    fn pow(self, rhs: u32) -> Self {
        Self {
            value: self.value.pow(rhs),
        }
    }
}

impl<'a> Pow<u32> for &'a Felt252 {
    type Output = Felt252;
    fn pow(self, rhs: u32) -> Self::Output {
        Self::Output {
            value: (&self.value).pow(rhs),
        }
    }
}

impl<'a> Pow<&'a Felt252> for &'a Felt252 {
    type Output = Felt252;
    fn pow(self, rhs: &'a Felt252) -> Self::Output {
        Self::Output {
            value: (&self.value).pow(&rhs.value),
        }
    }
}

impl Div for Felt252 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self {
            value: self.value / rhs.value,
        }
    }
}

impl<'a> Div for &'a Felt252 {
    type Output = Felt252;
    fn div(self, rhs: Self) -> Self::Output {
        Self::Output {
            value: &self.value / &rhs.value,
        }
    }
}

impl<'a> Div<Felt252> for &'a Felt252 {
    type Output = Felt252;
    fn div(self, rhs: Self::Output) -> Self::Output {
        Self::Output {
            value: &self.value / rhs.value,
        }
    }
}

impl Rem for Felt252 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        Self {
            value: self.value % rhs.value,
        }
    }
}

impl<'a> Rem<&'a Felt252> for Felt252 {
    type Output = Self;
    fn rem(self, rhs: &Self) -> Self {
        Self {
            value: self.value % &rhs.value,
        }
    }
}

impl Zero for Felt252 {
    fn zero() -> Self {
        Self {
            value: FeltBigInt::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }
}

impl One for Felt252 {
    fn one() -> Self {
        Self {
            value: FeltBigInt::one(),
        }
    }

    fn is_one(&self) -> bool {
        self.value.is_one()
    }
}

impl Bounded for Felt252 {
    fn min_value() -> Self {
        Self {
            value: FeltBigInt::min_value(),
        }
    }

    fn max_value() -> Self {
        Self {
            value: FeltBigInt::max_value(),
        }
    }
}

impl Num for Felt252 {
    type FromStrRadixErr = ParseFeltError;
    fn from_str_radix(string: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self {
            value: FeltBigInt::from_str_radix(string, radix)?,
        })
    }
}

impl Integer for Felt252 {
    fn div_floor(&self, rhs: &Self) -> Self {
        Self {
            value: self.value.div_floor(&rhs.value),
        }
    }

    fn div_rem(&self, other: &Self) -> (Self, Self) {
        let (div, rem) = self.value.div_rem(&other.value);
        (Self { value: div }, Self { value: rem })
    }

    fn divides(&self, other: &Self) -> bool {
        self.value.divides(&other.value)
    }

    fn gcd(&self, other: &Self) -> Self {
        Self {
            value: self.value.gcd(&other.value),
        }
    }

    fn is_even(&self) -> bool {
        self.value.is_even()
    }

    fn is_multiple_of(&self, other: &Self) -> bool {
        self.value.is_multiple_of(&other.value)
    }

    fn is_odd(&self) -> bool {
        self.value.is_odd()
    }

    fn lcm(&self, other: &Self) -> Self {
        Self {
            value: self.value.lcm(&other.value),
        }
    }

    fn mod_floor(&self, rhs: &Self) -> Self {
        Self {
            value: self.value.mod_floor(&rhs.value),
        }
    }
}

impl Signed for Felt252 {
    fn abs(&self) -> Self {
        Self {
            value: self.value.abs(),
        }
    }

    fn abs_sub(&self, other: &Self) -> Self {
        Self {
            value: self.value.abs_sub(&other.value),
        }
    }

    fn signum(&self) -> Self {
        Self {
            value: self.value.signum(),
        }
    }

    fn is_positive(&self) -> bool {
        self.value.is_positive()
    }

    fn is_negative(&self) -> bool {
        self.value.is_negative()
    }
}

impl Shl<u32> for Felt252 {
    type Output = Self;
    fn shl(self, rhs: u32) -> Self {
        Self {
            value: self.value << rhs,
        }
    }
}

impl<'a> Shl<u32> for &'a Felt252 {
    type Output = Felt252;
    fn shl(self, rhs: u32) -> Self::Output {
        Self::Output {
            value: &self.value << rhs,
        }
    }
}

impl Shl<usize> for Felt252 {
    type Output = Self;
    fn shl(self, rhs: usize) -> Self {
        Self {
            value: self.value << rhs,
        }
    }
}

impl<'a> Shl<usize> for &'a Felt252 {
    type Output = Felt252;
    fn shl(self, rhs: usize) -> Self::Output {
        Self::Output {
            value: &self.value << rhs,
        }
    }
}

impl Shr<u32> for Felt252 {
    type Output = Self;
    fn shr(self, rhs: u32) -> Self {
        Self {
            value: self.value >> rhs,
        }
    }
}

impl<'a> Shr<u32> for &'a Felt252 {
    type Output = Felt252;
    fn shr(self, rhs: u32) -> Self::Output {
        Self::Output {
            value: &self.value >> rhs,
        }
    }
}

impl ShrAssign<usize> for Felt252 {
    fn shr_assign(&mut self, rhs: usize) {
        self.value >>= rhs
    }
}

impl<'a> BitAnd for &'a Felt252 {
    type Output = Felt252;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self::Output {
            value: &self.value & &rhs.value,
        }
    }
}

impl<'a> BitAnd<&'a Felt252> for Felt252 {
    type Output = Self;
    fn bitand(self, rhs: &Self) -> Self {
        Self {
            value: self.value & &rhs.value,
        }
    }
}

impl<'a> BitAnd<Felt252> for &'a Felt252 {
    type Output = Felt252;
    fn bitand(self, rhs: Self::Output) -> Self::Output {
        Self::Output {
            value: &self.value & rhs.value,
        }
    }
}

impl<'a> BitOr for &'a Felt252 {
    type Output = Felt252;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self::Output {
            value: &self.value | &rhs.value,
        }
    }
}

impl<'a> BitXor for &'a Felt252 {
    type Output = Felt252;
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self::Output {
            value: &self.value ^ &rhs.value,
        }
    }
}

impl ToPrimitive for Felt252 {
    fn to_u128(&self) -> Option<u128> {
        self.value.to_u128()
    }

    fn to_u64(&self) -> Option<u64> {
        self.value.to_u64()
    }

    fn to_i64(&self) -> Option<i64> {
        self.value.to_i64()
    }
}

impl FromPrimitive for Felt252 {
    fn from_u64(n: u64) -> Option<Self> {
        FeltBigInt::from_u64(n).map(|n| Self { value: n })
    }

    fn from_i64(n: i64) -> Option<Self> {
        FeltBigInt::from_i64(n).map(|n| Self { value: n })
    }
}

impl fmt::Display for ParseFeltError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{ParseFeltError:?}")
    }
}

#[cfg(any(feature = "arbitrary", test))]
#[cfg_attr(feature = "lambdaworks-felt", path = "arbitrary_lambdaworks.rs")]
#[cfg_attr(not(feature = "lambdaworks-felt"), path = "arbitrary_bigint_felt.rs")]
/// [`proptest::arbitrary::Arbitrary`] implementation for [`Felt252`], and [`Strategy`] generating functions.
///
/// Not to be confused with [`arbitrary::Arbitrary`], which is also enabled by the _arbitrary_ feature.
pub mod arbitrary;
