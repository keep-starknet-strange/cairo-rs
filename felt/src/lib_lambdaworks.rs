use core::{
    convert::Into,
    fmt,
    iter::Sum,
    ops::{
        Add, AddAssign, BitAnd, BitOr, BitXor, Div, Mul, MulAssign, Neg, Rem, Shl, Shr, ShrAssign,
        Sub, SubAssign,
    },
};
use lambdaworks_math::{
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    },
    traits::ByteConversion,
    unsigned_integer::element::UnsignedInteger,
};
use lazy_static::lazy_static;
use num_bigint::{BigInt, BigUint, Sign, ToBigInt};
use num_integer::Integer;
use num_traits::{Bounded, FromPrimitive, Num, One, Pow, Signed, ToPrimitive, Zero};
#[cfg(feature = "parity-scale-codec")]
use parity_scale_codec::{Decode, Encode, Output};
use serde::{Deserialize, Serialize};

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};

use crate::{ParseFeltError, FIELD_HIGH, FIELD_LOW};

#[cfg(feature = "arbitrary")]
impl<'a> Arbitrary<'a> for Felt252 {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let num: BigUint = BigUint::arbitrary(u)?;
        Ok(Felt252::from(num))
    }
}

lazy_static! {
    pub static ref CAIRO_PRIME_BIGUINT: BigUint =
        (Into::<BigUint>::into(FIELD_HIGH) << 128) + Into::<BigUint>::into(FIELD_LOW);
    pub static ref SIGNED_FELT_MAX: BigUint = (&*CAIRO_PRIME_BIGUINT).shr(1_u32);
    pub static ref CAIRO_SIGNED_PRIME: BigInt = CAIRO_PRIME_BIGUINT
        .to_bigint()
        .expect("Conversion BigUint -> BigInt can't fail");
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

#[derive(Eq, Hash, PartialEq, Clone, Deserialize, Serialize)]
#[serde(from = "BigInt")]
#[serde(into = "BigInt")]
pub struct Felt252 {
    pub(crate) value: FieldElement<Stark252PrimeField>,
}

#[cfg(feature = "parity-scale-codec")]
impl Encode for Felt252 {
    fn encode_to<T: Output + ?Sized>(&self, dest: &mut T) {
        // Scale codec has its own way of representing data. The first 2 bits of the first
        // byte determines the type of data it is.
        // NOTE: The compact byte (first byte) is encoded in LE.
        // This implementation is highly inspired from this: https://github.com/polkascan/py-scale-codec
        if self <= &Self::from_u16(0b00111111).unwrap() {
            // 0b00: single-byte mode; upper six bits are the LE encoding of the value (valid only for values of 0-63).
            // Safe to unwrap because we checked the value right before.
            // We shift left by 2 bits to append the 0b00 prefix to the number in little endian.
            let encoded = self.to_u8().unwrap() << 2;
            dest.write(&[encoded]);
        } else if self <= &Self::from_u16(0b0011111111111111).unwrap() {
            // 0b01: two-byte mode: upper six bits and the following byte is the LE encoding of the value (valid only for values 64-(2**14-1)).
            // Safe to unwrap because we checked the value right before.
            // We shift left by 2 bits to have 0b00 at the end of the number then we apply | 0b01 to append the 0b01 prefix to the number in little endian.
            let encoded = self.to_u16().unwrap() << 2 | 0b01;
            dest.write(&encoded.to_le_bytes());
        } else if self <= &Self::from_u32(0b00111111111111111111111111111111).unwrap() {
            // 0b10: four-byte mode: upper six bits and the following three bytes are the LE encoding of the value (valid only for values (2**14)-(2**30-1)).
            // Safe to unwrap because we checked the value right before.
            // We shift left by 2 bits to have 0b00 at the end of the number then we apply | 0b10 to append the 0b10 prefix to the number in little endian.
            let encoded = self.to_u32().unwrap() << 2 | 0b10;
            dest.write(&encoded.to_le_bytes());
        } else {
            // 0b11: Big-integer mode: The upper six bits are the number of bytes following, plus four.
            // The value is contained, LE encoded, in the bytes following.
            // The final (most significant) byte must be non-zero. Valid only for values (2**30)-(2**536-1).
            let mut bytes = self
                .to_le_bytes()
                .into_iter()
                .rev()
                .skip_while(|&x| x == 0)
                .collect::<Vec<u8>>();
            bytes.reverse();
            let mut encoded_value = vec![((bytes.len() - 4) << 2 | 0b11).to_le_bytes()[0]];
            // For example for val = 100000000000000 we would have:
            // first byte : 0b11011000 == 11_u8
            // All the bytes [11, 0, 64, 122, 16, 243, 90]
            // Hex: 0x0b00407a10f35a
            // see this example: https://docs.substrate.io/reference/scale-codec/
            // To verify the result you can run this python snippet:
            // hex(int.from_bytes(bytes([11, 0, 64, 122, 16, 243, 90]), 'big'))
            encoded_value.extend(bytes.iter());
            dest.write(&encoded_value)
        }
    }
}

#[cfg(feature = "parity-scale-codec")]
impl Decode for Felt252 {
    fn decode<I: parity_scale_codec::Input>(
        input: &mut I,
    ) -> Result<Self, parity_scale_codec::Error> {
        let compact_byte = input.read_byte()?;
        let compact_length = match compact_byte % 4 {
            0 => 1,
            1 => 2,
            2 => 4,
            _ => 5 + (compact_byte - 3) / 4,
        };
        Ok(if compact_length == 1 {
            Self::from_u8(compact_byte.to_be()).unwrap() / Self::from_u8(4).unwrap()
        } else if [2, 4].contains(&compact_length) {
            let mut buffer = vec![0; compact_length as usize - 1];
            input.read(&mut buffer)?;
            let mut bytes = [0; 4];
            bytes[0] = compact_byte;
            bytes[1..compact_length as usize].copy_from_slice(&buffer);
            Self::from_u32(u32::from_le_bytes(bytes) / 4).unwrap()
        } else {
            let mut buffer = vec![0; compact_length as usize - 1];
            input.read(&mut buffer)?;
            let mut bytes = [0u8; 32];
            bytes[..compact_length as usize - 1].copy_from_slice(&buffer);
            Self {
                // It was encoded before it can be decoded.
                value: FieldElement::<Stark252PrimeField>::from_bytes_le(&bytes).unwrap(),
            }
        })
    }
}

// TODO: remove and change for transformation + compare
impl PartialOrd for Felt252 {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// TODO: remove and change for transformation + compare
// Also, maybe this could be changed to compare against zero without changing montgomeryness
impl Ord for Felt252 {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.value
            .representative()
            .cmp(&other.value.representative())
    }
}

impl Default for Felt252 {
    fn default() -> Self {
        Self {
            value: FieldElement::zero(),
        }
    }
}

macro_rules! from_num {
    ($type:ty, $cast:ty) => {
        impl From<$type> for Felt252 {
            fn from(value: $type) -> Self {
                let uplifted: $cast = value as $cast;
                uplifted.into()
            }
        }
    };
}

from_num!(isize, i64);
from_num!(i8, i64);
from_num!(i16, i64);
from_num!(i32, i64);

// TODO: move to upstream?
impl From<i64> for Felt252 {
    fn from(value: i64) -> Self {
        let value = if !value.is_negative() {
            FieldElement::new(UnsignedInteger::from_u64(value as u64))
        } else {
            let abs_minus_one = UnsignedInteger::from_u64(-(value + 1) as u64);
            FieldElement::zero() - FieldElement::one() - FieldElement::new(abs_minus_one)
        };
        Self { value }
    }
}

// TODO: move to upstream?
impl From<i128> for Felt252 {
    fn from(value: i128) -> Self {
        let value = if !value.is_negative() {
            FieldElement::new(UnsignedInteger::from_u128(value as u128))
        } else {
            let abs_minus_one = UnsignedInteger::from_u128(-(value + 1) as u128);
            FieldElement::zero() - FieldElement::one() - FieldElement::new(abs_minus_one)
        };
        Self { value }
    }
}

from_num!(usize, u64);
from_num!(u8, u64);
from_num!(u16, u64);
from_num!(u32, u64);

// TODO: move to upstream?
impl From<u64> for Felt252 {
    fn from(value: u64) -> Self {
        let value = FieldElement::new(UnsignedInteger::from_u64(value));
        Self { value }
    }
}

// TODO: move to upstream?
impl From<u128> for Felt252 {
    fn from(value: u128) -> Self {
        let value = FieldElement::new(UnsignedInteger::from_u128(value));
        Self { value }
    }
}

impl From<bool> for Felt252 {
    fn from(flag: bool) -> Self {
        if flag {
            Self::one()
        } else {
            Self::zero()
        }
    }
}

impl From<BigUint> for Felt252 {
    fn from(mut value: BigUint) -> Self {
        if value >= *CAIRO_PRIME_BIGUINT {
            value = value.mod_floor(&CAIRO_PRIME_BIGUINT);
        }
        let mut limbs = [0; 4];
        for (i, l) in (0..4).rev().zip(value.iter_u64_digits()) {
            limbs[i] = l;
        }
        let value = FieldElement::new(UnsignedInteger::from_limbs(limbs));
        Self { value }
    }
}

impl From<&BigUint> for Felt252 {
    fn from(value: &BigUint) -> Self {
        if value >= &CAIRO_PRIME_BIGUINT {
            Self::from(value.clone())
        } else {
            let mut limbs = [0; 4];
            for (i, l) in (0..4).rev().zip(value.iter_u64_digits()) {
                limbs[i] = l;
            }
            let value = FieldElement::new(UnsignedInteger::from_limbs(limbs));
            Self { value }
        }
    }
}

// NOTE: used for deserialization
impl From<BigInt> for Felt252 {
    fn from(value: BigInt) -> Self {
        let val = value.mod_floor(&CAIRO_PRIME_BIGUINT.to_bigint().expect("cannot fail"));
        let mut limbs = [0; 4];
        for (i, l) in (0..4).rev().zip(val.iter_u64_digits()) {
            limbs[i] = l;
        }
        let value = FieldElement::new(UnsignedInteger::from_limbs(limbs));
        Self { value }
    }
}

// NOTE: used for serialization
impl From<Felt252> for BigInt {
    fn from(value: Felt252) -> Self {
        value.to_bigint()
    }
}

impl Felt252 {
    pub fn new<T: Into<Felt252>>(value: T) -> Self {
        value.into()
    }

    pub fn iter_u64_digits(&self) -> impl Iterator<Item = u64> {
        self.value.representative().limbs.into_iter().rev()
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn to_bytes_be(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    pub fn to_le_bytes(&self) -> [u8; 32] {
        // TODO: upstream should return array
        let mut bytes = [0; 32];
        let digits = self.to_le_digits();
        for (i, d) in digits.into_iter().enumerate() {
            let idx = i * 8;
            bytes[idx..(idx + 8)].copy_from_slice(&d.to_le_bytes());
        }
        bytes
    }

    pub fn to_be_bytes(&self) -> [u8; 32] {
        // TODO: upstream should return array
        let mut bytes = [0; 32];
        let digits = self.to_be_digits();
        for (i, d) in digits.into_iter().enumerate() {
            let idx = i * 8;
            bytes[idx..(idx + 8)].copy_from_slice(&d.to_be_bytes());
        }
        bytes
    }

    pub fn to_le_digits(&self) -> [u64; 4] {
        let mut rep = self.value.representative();
        rep.limbs.reverse();
        rep.limbs
    }

    pub fn to_be_digits(&self) -> [u64; 4] {
        self.value.representative().limbs
    }

    pub fn parse_bytes(bytes: &[u8], radix: u32) -> Option<Self> {
        Some(BigInt::parse_bytes(bytes, radix)?.into())
    }

    pub fn from_bytes_be(bytes: &[u8]) -> Self {
        // TODO: use upstream's version when it's more lenient
        Self::from(BigUint::from_bytes_be(bytes))
    }

    pub fn from_bytes_le(bytes: &[u8]) -> Self {
        // TODO: use upstream's version when it's more lenient
        Self::from(BigUint::from_bytes_le(bytes))
    }

    pub fn from_bytes_ne(bytes: &[u8]) -> Self {
        // Call either version depending on target endianness
        #[cfg(target_endian = "little")]
        let res = Self::from_bytes_le(bytes);
        #[cfg(target_endian = "big")]
        let res = Self::from_bytes_be(bytes);
        res
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn to_str_radix(&self, radix: u32) -> String {
        if radix == 16 {
            let mut res = format!("{}", self.value);
            res.replace_range(..2, "");
            res
        } else {
            self.to_biguint().to_str_radix(radix)
        }
    }

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
    pub fn to_signed_felt(&self) -> BigInt {
        let biguint = self.to_biguint();
        if biguint > *SIGNED_FELT_MAX {
            BigInt::from_biguint(num_bigint::Sign::Minus, &*CAIRO_PRIME_BIGUINT - &biguint)
        } else {
            biguint.to_bigint().expect("cannot fail")
        }
    }

    // Converts [`Felt252`]'s representation directly into a [`BigInt`].
    // Equivalent to doing felt.to_biguint().to_bigint().
    pub fn to_bigint(&self) -> BigInt {
        BigInt::from_biguint(Sign::Plus, self.to_biguint())
    }

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
    pub fn to_biguint(&self) -> BigUint {
        let big_digits = self
            .iter_u64_digits()
            .flat_map(|limb| [limb as u32, (limb >> 32) as u32])
            .collect();
        BigUint::new(big_digits)
    }

    pub fn sqrt(&self) -> Self {
        // Safety: must be called with residual
        let (root_1, root_2) = self.value.sqrt().unwrap();
        let value = FieldElement::new(root_1.representative().min(root_2.representative()));
        Self { value }
    }

    pub fn bits(&self) -> u64 {
        // TODO: move upstream
        let rep = self.value.representative();
        match rep.limbs {
            [0, 0, 0, 0] => 0,
            [0, 0, 0, l0] => u64::BITS - l0.leading_zeros(),
            [0, 0, l1, _] => 2 * u64::BITS - l1.leading_zeros(),
            [0, l2, _, _] => 3 * u64::BITS - l2.leading_zeros(),
            [l3, _, _, _] => 4 * u64::BITS - l3.leading_zeros(),
        }
        .into()
    }

    pub fn prime() -> BigUint {
        CAIRO_PRIME_BIGUINT.clone()
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
            value: self.value + rhs.value,
        }
    }
}

impl<'a> Add<&'a Felt252> for Felt252 {
    type Output = Self;
    fn add(self, rhs: &Self) -> Self::Output {
        Self::Output {
            value: self.value + rhs.value,
        }
    }
}

impl Add<u32> for Felt252 {
    type Output = Self;
    fn add(self, rhs: u32) -> Self {
        let rhs = UnsignedInteger::from_u64(rhs.into());
        Self {
            value: self.value + FieldElement::new(rhs),
        }
    }
}

impl Add<usize> for Felt252 {
    type Output = Self;
    fn add(self, rhs: usize) -> Self {
        let rhs = UnsignedInteger::from_u64(rhs as u64);
        Self {
            value: self.value + FieldElement::new(rhs),
        }
    }
}

impl<'a> Add<usize> for &'a Felt252 {
    type Output = Felt252;
    fn add(self, rhs: usize) -> Self::Output {
        let rhs = UnsignedInteger::from_u64(rhs as u64);
        Self::Output {
            value: self.value + FieldElement::new(rhs),
        }
    }
}

impl Add<u64> for &Felt252 {
    type Output = Felt252;
    fn add(self, rhs: u64) -> Self::Output {
        let rhs = UnsignedInteger::from_u64(rhs);
        Self::Output {
            value: self.value + FieldElement::new(rhs),
        }
    }
}

// TODO: verify if this optimization causes a speed-up in the current implementation.
// This is special cased and optimized compared to the obvious implementation
// due to `pc_update` relying on this, which makes it a major bottleneck for
// execution. Testing for this function is extensive, comprised of explicit
// edge and special cases testing and property tests, all comparing to the
// more intuitive `(rhs + self).to_u64()` implementation.
impl Add<&Felt252> for u64 {
    type Output = Option<u64>;

    fn add(self, rhs: &Felt252) -> Option<u64> {
        const PRIME_DIGITS_BE_HI: [u64; 3] =
            [0x0800000000000011, 0x0000000000000000, 0x0000000000000000];
        const PRIME_MINUS_U64_MAX_DIGITS_BE_HI: [u64; 3] =
            [0x0800000000000010, 0xffffffffffffffff, 0xffffffffffffffff];

        // Match with the 64 bits digits in big-endian order to
        // characterize how the sum will behave.
        match rhs.to_be_digits() {
            // All digits are `0`, so the sum is simply `self`.
            [0, 0, 0, 0] => Some(self),
            // A single digit means this is effectively the sum of two `u64` numbers.
            [0, 0, 0, low] => self.checked_add(low),
            // Now we need to compare the 3 most significant digits.
            // There are two relevant cases from now on, either `rhs` behaves like a
            // substraction of a `u64` or the result of the sum falls out of range.

            // The 3 MSB only match the prime for Felt252::max_value(), which is -1
            // in the signed field, so this is equivalent to substracting 1 to `self`.
            [hi @ .., _] if hi == PRIME_DIGITS_BE_HI => self.checked_sub(1),

            // For the remaining values between `[-u64::MAX..0]` (where `{0, -1}` have
            // already been covered) the MSB matches that of `PRIME - u64::MAX`.
            // Because we're in the negative number case, we count down. Because `0`
            // and `-1` correspond to different MSBs, `0` and `1` in the LSB are less
            // than `-u64::MAX`, the smallest value we can add to (read, substract its
            // magnitude from) a `u64` number, meaning we exclude them from the valid
            // case.
            // For the remaining range, we take the absolute value module-2 while
            // correcting by substracting `1` (note we actually substract `2` because
            // the absolute value itself requires substracting `1`.
            [hi @ .., low] if hi == PRIME_MINUS_U64_MAX_DIGITS_BE_HI && low >= 2 => {
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
        // TODO: optimize and move upstream
        self.value += rhs.value;
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
            value: self.value - rhs.value,
        }
    }
}

impl<'a> Sub<&'a Felt252> for Felt252 {
    type Output = Self;
    fn sub(self, rhs: &Self) -> Self {
        Self {
            value: self.value - rhs.value,
        }
    }
}

// a - b = a + (-b), but clippy doesn't know that
#[allow(clippy::suspicious_arithmetic_impl)]
impl Sub<&Felt252> for usize {
    type Output = Felt252;
    fn sub(self, rhs: &Self::Output) -> Self::Output {
        let neg = Self::Output {
            value: (&rhs.value).neg(),
        };
        neg + self
    }
}

impl SubAssign for Felt252 {
    fn sub_assign(&mut self, rhs: Self) {
        // TODO: optimize and move to upstream
        self.value = self.value - rhs.value
    }
}

impl<'a> SubAssign<&'a Felt252> for Felt252 {
    fn sub_assign(&mut self, rhs: &Self) {
        // TODO: optimize and move to upstream
        self.value = self.value - rhs.value
    }
}

impl Sub<u32> for Felt252 {
    type Output = Self;
    fn sub(self, rhs: u32) -> Self {
        let value = self.value - FieldElement::new(UnsignedInteger::from_u64(rhs as u64));
        Self { value }
    }
}

impl<'a> Sub<u32> for &'a Felt252 {
    type Output = Felt252;
    fn sub(self, rhs: u32) -> Self::Output {
        self.clone() - rhs
    }
}

impl Sub<usize> for Felt252 {
    type Output = Self;
    fn sub(self, rhs: usize) -> Self {
        let value = self.value - FieldElement::new(UnsignedInteger::from_u64(rhs as u64));
        Self { value }
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
            value: self.value * rhs.value,
        }
    }
}

impl<'a> Mul<&'a Felt252> for Felt252 {
    type Output = Self;
    fn mul(self, rhs: &Self) -> Self {
        Self {
            value: self.value * rhs.value,
        }
    }
}

impl<'a> MulAssign<&'a Felt252> for Felt252 {
    fn mul_assign(&mut self, rhs: &Self) {
        self.value = self.value * rhs.value;
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
            value: self.value.pow(rhs),
        }
    }
}

impl<'a> Pow<&'a Felt252> for &'a Felt252 {
    type Output = Felt252;
    fn pow(self, rhs: &'a Felt252) -> Self::Output {
        Self::Output {
            value: self.value.pow(rhs.value.representative()),
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
            value: self.value / rhs.value,
        }
    }
}

impl<'a> Div<Felt252> for &'a Felt252 {
    type Output = Felt252;
    fn div(self, rhs: Self::Output) -> Self::Output {
        Self::Output {
            value: self.value / rhs.value,
        }
    }
}

impl Rem for Felt252 {
    type Output = Self;
    fn rem(self, _rhs: Self) -> Self {
        Self::zero()
    }
}

impl<'a> Rem<&'a Felt252> for Felt252 {
    type Output = Self;
    fn rem(self, _rhs: &Self) -> Self {
        Self::zero()
    }
}

impl Zero for Felt252 {
    fn zero() -> Self {
        Self {
            value: FieldElement::from_raw(&Stark252PrimeField::ZERO),
        }
    }

    fn is_zero(&self) -> bool {
        self.value == FieldElement::from_raw(&Stark252PrimeField::ZERO)
    }
}

impl One for Felt252 {
    fn one() -> Self {
        let value = FieldElement::from_raw(&Stark252PrimeField::ONE);
        Self { value }
    }

    fn is_one(&self) -> bool {
        self.value == FieldElement::from_raw(&Stark252PrimeField::ONE)
    }
}

impl Bounded for Felt252 {
    fn min_value() -> Self {
        Self {
            value: FieldElement::zero(),
        }
    }

    fn max_value() -> Self {
        Self {
            value: FieldElement::zero() - FieldElement::one(),
        }
    }
}

impl Num for Felt252 {
    type FromStrRadixErr = ParseFeltError;
    fn from_str_radix(string: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let res = if radix == 16 {
            let value = FieldElement::from_hex(string).map_err(|_| ParseFeltError)?;
            Self { value }
        } else {
            let biguint = BigInt::from_str_radix(string, radix).map_err(|_| ParseFeltError)?;
            biguint.into()
        };
        Ok(res)
    }
}

impl Integer for Felt252 {
    fn div_floor(&self, rhs: &Self) -> Self {
        let (d, _) = self.div_rem(rhs);
        d
    }

    fn mod_floor(&self, rhs: &Self) -> Self {
        let (_, m) = self.div_rem(rhs);
        m
    }

    fn div_rem(&self, other: &Self) -> (Self, Self) {
        let (div, rem) = self.to_biguint().div_mod_floor(&other.to_biguint());
        (Self::from(div), Self::from(rem))
    }

    // NOTE: we overload because the default impl calls div_floor AND mod_floor.
    fn div_mod_floor(&self, rhs: &Self) -> (Self, Self) {
        // NOTE: for positive integers, to floor and truncate is the same, so div_rem == div_mod_floor.
        self.div_rem(rhs)
    }

    fn divides(&self, _other: &Self) -> bool {
        !self.is_zero()
    }

    fn gcd(&self, other: &Self) -> Self {
        Self::from(self.to_biguint().gcd(&other.to_biguint()))
    }

    fn is_even(&self) -> bool {
        self.value.representative().limbs[3] & 1 == 0
    }

    fn is_multiple_of(&self, other: &Self) -> bool {
        !other.is_zero()
    }

    fn is_odd(&self) -> bool {
        !self.is_even()
    }

    fn lcm(&self, other: &Self) -> Self {
        self.max(other).clone()
    }
}

impl Signed for Felt252 {
    fn abs(&self) -> Self {
        self.clone()
    }

    fn abs_sub(&self, other: &Self) -> Self {
        self.max(other) - self.min(other)
    }

    fn signum(&self) -> Self {
        if self.is_zero() {
            Self::zero()
        } else {
            Self::one()
        }
    }

    fn is_positive(&self) -> bool {
        !self.is_zero()
    }

    fn is_negative(&self) -> bool {
        false
    }
}

// -------------------
// Bit-wise operations
// NOTE: these do bit shifting on the representative

impl Shl<u32> for Felt252 {
    type Output = Self;
    fn shl(self, rhs: u32) -> Self {
        &self << rhs
    }
}

impl<'a> Shl<u32> for &'a Felt252 {
    type Output = Felt252;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn shl(self, rhs: u32) -> Self::Output {
        // TODO: upstream should be able to receive usize
        Felt252::from(2).pow(rhs) * self
    }
}

impl Shl<usize> for Felt252 {
    type Output = Self;
    fn shl(self, rhs: usize) -> Self {
        &self << rhs
    }
}

impl<'a> Shl<usize> for &'a Felt252 {
    type Output = Felt252;
    fn shl(self, rhs: usize) -> Self::Output {
        self << (rhs as u32)
    }
}

impl Shr<u32> for Felt252 {
    type Output = Self;
    fn shr(self, rhs: u32) -> Self {
        &self >> rhs
    }
}

impl<'a> Shr<u32> for &'a Felt252 {
    type Output = Felt252;
    fn shr(self, rhs: u32) -> Self::Output {
        self >> (rhs as usize)
    }
}

impl Shr<usize> for Felt252 {
    type Output = Felt252;
    fn shr(self, rhs: usize) -> Self::Output {
        &self >> rhs
    }
}

impl<'a> Shr<usize> for &'a Felt252 {
    type Output = Felt252;
    fn shr(self, rhs: usize) -> Self::Output {
        // TODO: upstream should do this check
        if rhs >= 64 * 4 {
            Felt252::zero()
        } else {
            let value = FieldElement::new(self.value.representative() >> rhs);
            Self::Output { value }
        }
    }
}

impl ShrAssign<usize> for Felt252 {
    fn shr_assign(&mut self, rhs: usize) {
        // TODO: optimize and move upstream
        *self = self.clone() >> rhs;
    }
}

impl<'a> BitAnd for &'a Felt252 {
    type Output = Felt252;
    fn bitand(self, rhs: Self) -> Self::Output {
        self.clone() & rhs
    }
}

impl<'a> BitAnd<&'a Felt252> for Felt252 {
    type Output = Self;
    fn bitand(self, rhs: &Self) -> Self {
        rhs & self
    }
}

impl<'a> BitAnd<Felt252> for &'a Felt252 {
    type Output = Felt252;
    fn bitand(self, rhs: Self::Output) -> Self::Output {
        let a = self.value.representative();
        let b = rhs.value.representative();

        let value = FieldElement::new(a & b);
        Self::Output { value }
    }
}

impl<'a> BitOr for &'a Felt252 {
    type Output = Felt252;
    fn bitor(self, rhs: Self) -> Self::Output {
        let a = self.value.representative();
        let b = rhs.value.representative();

        let value = FieldElement::new(a | b);
        Self::Output { value }
    }
}

impl<'a> BitXor for &'a Felt252 {
    type Output = Felt252;
    fn bitxor(self, rhs: Self) -> Self::Output {
        let a = self.value.representative();
        let b = rhs.value.representative();

        let value = FieldElement::new(a ^ b);
        Self::Output { value }
    }
}

// TODO: move to upstream
impl ToPrimitive for Felt252 {
    fn to_u128(&self) -> Option<u128> {
        match self.value.representative().limbs {
            [0, 0, high, low] => Some(((high as u128) << 64) | low as u128),
            _ => None,
        }
    }

    fn to_u64(&self) -> Option<u64> {
        match self.value.representative().limbs {
            [0, 0, 0, val] => Some(val),
            _ => None,
        }
    }

    fn to_i64(&self) -> Option<i64> {
        // NOTE: result can't be negative
        self.to_u64().as_ref().and_then(u64::to_i64)
    }
}

impl FromPrimitive for Felt252 {
    fn from_u64(n: u64) -> Option<Self> {
        Some(Felt252 {
            value: FieldElement::from(n),
        })
    }

    fn from_i64(n: i64) -> Option<Self> {
        let res = (!n.is_negative()).then(|| FieldElement::from(n as u64));
        res.map(|value| Felt252 { value })
    }
}

impl fmt::Display for Felt252 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_biguint().to_str_radix(10))
    }
}

impl fmt::Debug for Felt252 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_biguint().to_str_radix(10))
    }
}

#[cfg(test)]
mod test {
    use core::cmp;

    use super::*;
    use crate::{arbitrary::nonzero_felt252, PRIME_STR};
    use num_integer::Integer;
    use rstest::rstest;

    use num_integer::Integer;
    use proptest::prelude::*;
    use rstest::rstest;

    proptest! {
        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property-based test that ensures, for 100 felt values that are randomly generated
        // each time tests are run, that a new felt doesn't fall outside the range [0, p].
        // In this and some of the following tests, The value of {x} can be either [0] or a
        // very large number, in order to try to overflow the value of {p} and thus ensure the
        // modular arithmetic is working correctly.
        fn new_in_range_be(ref x in any::<[u8; 40]>()) {
            let x = Felt252::from_bytes_be(x);
            let p = &BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();
            prop_assert!(&x.to_biguint() < p);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        fn new_in_range_le(ref x in any::<[u8; 40]>()) {
            let x = Felt252::from_bytes_le(x);
            let p = &BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();
            prop_assert!(&x.to_biguint() < p);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        fn from_bytes_be(high: u128, low: u128) {
            let expected = (Felt252::from(high) << 128_usize) + Felt252::from(low);
            let mut bytes = [0; 32];
            // big-endian order: [ high, low ]
            bytes[..16].copy_from_slice(&high.to_be_bytes());
            bytes[16..].copy_from_slice(&low.to_be_bytes());
            let got = Felt252::from_bytes_be(&bytes);
            prop_assert_eq!(got, expected);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        fn from_bytes_le(high: u128, low: u128) {
            let expected = (Felt252::from(high) << 128_usize) + Felt252::from(low);
            let mut bytes = [0; 32];
            // little-endian order: [ low, high ]
            bytes[..16].copy_from_slice(&low.to_le_bytes());
            bytes[16..].copy_from_slice(&high.to_le_bytes());
            let got = Felt252::from_bytes_le(&bytes);
            prop_assert_eq!(got, expected);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        fn to_be_bytes(ref x in any::<Felt252>()) {
            let bytes = x.to_be_bytes();
            let y = &Felt252::from_bytes_be(&bytes);
            prop_assert_eq!(x, y);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        fn to_le_bytes(ref x in any::<Felt252>()) {
            let mut bytes = x.to_le_bytes();
            // Convert to big endian for test
            bytes.reverse();
            let y = &Felt252::from_bytes_be(&bytes);
            prop_assert_eq!(x, y);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        fn to_le_digits(ref x in any::<Felt252>()) {
            let digits: [u64; 4] = x.to_le_digits();
            let mut bytes: Vec<_> = digits
                .into_iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            // Convert to big endian for test
            bytes.reverse();
            let y = &Felt252::from_bytes_be(&bytes);
            prop_assert_eq!(x, y);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        fn to_u128_ok(x in any::<u128>()) {
            let y = Felt252::from(x);
            let y = y.to_u128();
            prop_assert_eq!(Some(x), y);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        fn to_u128_out_of_range(x in nonzero_felt252()) {
            let y = x + Felt252::from(u128::MAX);
            let y = y.to_u128();
            prop_assert_eq!(None, y);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property-based test that ensures, for 100 felt values that are randomly
        // generated each time tests are run, that a felt created using Felt252::from_bytes_be doesn't
        // fall outside the range [0, p].
        // In this and some of the following tests, The value of {x} can be either [0] or a very large number,
        // in order to try to overflow the value of {p} and thus ensure the modular arithmetic is working correctly.
        fn from_bytes_be_in_range(ref x in any::<[u8; 40]>()) {
            let x = Felt252::from_bytes_be(x);
            let max_felt = Felt252::max_value();
            prop_assert!(x <= max_felt);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property-based test that ensures, for 100 felt values that are randomly generated each time
        // tests are run, that the negative of a felt doesn't fall outside the range [0, p].
        fn neg_in_range(x in any::<Felt252>()) {
            let p = &BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();

            let neg = -x.clone();
            let as_uint = &neg.to_biguint();
            prop_assert!(as_uint < p);

            // test reference variant
            let neg = -&x;
            let as_uint = &neg.to_biguint();
            prop_assert!(as_uint < p);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property-based test that ensures, for 100 {x} and {y} values that are randomly generated
        // each time tests are run, that a subtraction between two felts {x} and {y} and doesn't fall
        // outside the range [0, p]. The values of {x} and {y} can be either [0] or a very large number.
        fn sub(ref x in any::<Felt252>(), ref y in any::<Felt252>()) {
            let (x_int, y_int) = (&x.to_biguint(), &y.to_biguint());
            let p = &BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();

            let sub_xy = x - y;
            prop_assert!(&sub_xy.to_biguint() < p);
            prop_assert_eq!(Felt252::from(p + x_int - y_int), sub_xy);

            let sub_yx = y - x;
            prop_assert!(&sub_yx.to_biguint() < p);
            prop_assert_eq!(Felt252::from(p + y_int - x_int), sub_yx);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property-based test that ensures, for 100 {x} and {y} values that are randomly generated
        // each time tests are run, that a subtraction with assignment between two felts {x} and {y}
        // and doesn't fall outside the range [0, p]. The values of {x} and {y} can be either [0] or a very large number.
        fn sub_assign_in_range(mut x in any::<Felt252>(), y in any::<Felt252>()) {
            let p = &BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();

            x -= y.clone();
            let as_uint = &x.to_biguint();
            prop_assert!(as_uint < p, "{}", as_uint);

            // test reference variant
            x -= &y;
            let as_uint = &x.to_biguint();
            prop_assert!(as_uint < p, "{}", as_uint);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property-based test that ensures, for 100 {x} and {y} values that are randomly
        // generated each time tests are run, that a multiplication between two felts {x}
        // and {y} and doesn't fall outside the range [0, p]. The values of {x} and {y}
        // can be either [0] or a very large number.
        fn mul(ref x in any::<Felt252>(), ref y in any::<Felt252>()) {
            let xy_int = x.to_biguint() * y.to_biguint();

            let p = &BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();

            let (xy, yx) = (x * y, y * x);
            prop_assert_eq!(&xy, &yx);
            prop_assert_eq!(xy.to_biguint(), xy_int.mod_floor(p));
            prop_assert!(&xy.to_biguint() < p);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property-based test that ensures, for 100 pairs of {x} and {y} values that
        // are randomly generated each time tests are run, that a multiplication with
        // assignment between two felts {x} and {y} and doesn't fall outside the range [0, p].
        // The values of {x} and {y} can be either [0] or a very large number.
        fn mul_assign_in_range(mut x in any::<Felt252>(), y in any::<Felt252>()) {
            let p = &BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();

            x *= &y;
            let as_uint = &x.to_biguint();
            prop_assert!(as_uint < p, "{}", as_uint);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property-based test that ensures, for 100 pairs of {x} and {y} values that are
        // randomly generated each time tests are run, that the result of the division of
        // {x} by {y} is the inverse multiplicative of {x} --that is, multiplying the result
        // by {y} returns the original number {x}. The values of {x} and {y} can be either
        // [0] or a very large number.
        fn div_is_mul_inv(ref x in any::<Felt252>(), ref y in nonzero_felt252()) {
            let p = &BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();
            prop_assume!(!y.is_zero());

            let q = x / y;
            let as_uint = &q.to_biguint();
            prop_assert!(as_uint < p, "{}", as_uint);
            prop_assert_eq!(&(q * y), x);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property-based test that ensures, for 100 {value}s that are randomly generated
        // each time tests are run, that performing a bit shift to the left by {shift_amount}
        // of bits (between 0 and 999) returns a result that is inside of the range [0, p].
        fn shift_left_in_range(value in any::<Felt252>(), shift_amount in 0..1000_u32) {
            let p = &BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();

            let result = (value.clone() << shift_amount).to_biguint();
            prop_assert!(&result < p);

            let result = (&value << shift_amount).to_biguint();
            prop_assert!(&result < p);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        fn shift_left_equals_old_shl(value in any::<Felt252>(), shift_amount in 0..1000_u32) {
            let expected = (value.to_biguint() << shift_amount).mod_floor(&Felt252::prime());

            let result = (&value << shift_amount).to_biguint();
            prop_assert_eq!(&result, &expected);

            let result = (value << shift_amount).to_biguint();
            prop_assert_eq!(&result, &expected);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property-based test that ensures, for 100 {value}s that are randomly
        // generated each time tests are run, that performing a bit shift to the right
        // by {shift_amount} of bits (between 0 and 999) returns a result that is inside of the range [0, p].
        fn shift_right_in_range(value in any::<Felt252>(), shift_amount in 0..1000_u32){
            let result = (value >> shift_amount).to_biguint();
            let p = &BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();
            prop_assert!(&result < p);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property-based test that ensures, for 100 {value}s that are randomly generated
        // each time tests are run, that performing a bit shift to the right by {shift_amount}
        // of bits (between 0 and 999), with assignment, returns a result that is inside of the range [0, p].
        // "With assignment" means that the result of the operation is autommatically assigned
        // to the variable value, replacing its previous content.
        fn shift_right_assign_in_range(mut value in any::<Felt252>(), shift_amount in 0..1000_usize) {
            let p = BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();
            value >>= shift_amount;
            prop_assert!(value.to_biguint() < p);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        fn shift_right_equals_old_shr(value in any::<Felt252>(), shift_amount in 0..1000_u32) {
            let expected = (value.to_biguint() >> shift_amount).mod_floor(&Felt252::prime());

            let result = (&value >> shift_amount).to_biguint();
            prop_assert_eq!(&result, &expected);

            let result = (value >> shift_amount).to_biguint();
            prop_assert_eq!(&result, &expected);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property based test that ensures, for 100 pairs of values {x} and {y}
        // generated at random each time tests are run, that performing a BitAnd
        // operation between them returns a result that is inside of the range [0, p].
        fn bitand_in_range(x in any::<Felt252>(), y in any::<Felt252>()){
            let p = BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();
            let result = x & &y;
            result.to_biguint();
            prop_assert!(result.to_biguint() < p);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property based test that ensures, for 100 pairs of values {x} and {y}
        // generated at random each time tests are run, that performing a BitOr
        // operation between them returns a result that is inside of the range [0, p].
        fn bitor_in_range(x in any::<Felt252>(), y in any::<Felt252>()){
            let p = BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();
            let result = &x | &y;
            prop_assert!(result.to_biguint() < p);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property based test that ensures, for 100 pairs of values {x} and {y}
        // generated at random each time tests are run, that performing a BitXor
        // operation between them returns a result that is inside of the range [0, p].
        fn bitxor_in_range(x in any::<Felt252>(), y in any::<Felt252>()){
            let p = BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();
            let result = &x ^ &y;
            prop_assert!(result.to_biguint() < p);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property-based test that ensures, for 100 values {x} that are randomly
        // generated each time tests are run, that raising {x} to the {y}th power
        // returns a result that is inside of the range [0, p].
        fn pow_in_range(base in any::<Felt252>(), exp in 0..100_u32){
            let p = &BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();

            let result = Pow::pow(base.clone(), exp);
            let as_uint = &result.to_biguint();
            prop_assert!(as_uint < p, "{}", as_uint);

            // test reference variant
            let result = Pow::pow(&base, exp);
            let as_uint = &result.to_biguint();
            prop_assert!(as_uint < p, "{}", as_uint);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property-based test that ensures, for 100 values {x} that are randomly
        // generated each time tests are run, that raising {x} to the {y}th power
        // returns a result that is inside of the range [0, p].
        fn pow_felt_in_range(base in any::<Felt252>(), exponent in any::<Felt252>()){
            let p = BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();

            let result = Pow::pow(&base, &exponent);
            let as_uint = result.to_biguint();
            prop_assert!(as_uint < p, "{}", as_uint);

            // test reference variant
            let result: Felt252 = Pow::pow(&base, &exponent);
            let as_uint = result.to_biguint();
            prop_assert!(as_uint < p, "{}", as_uint);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property based test that ensures, for 100 pairs of values {x} and {y}
        // generated at random each time tests are run, that performing a Sum operation
        // between them returns a result that is inside of the range [0, p].
        fn sum_in_range(x in any::<Felt252>(), y in any::<Felt252>()){
            let p = &BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();

            let result = x + y;
            let as_uint = &result.to_biguint();
            prop_assert!(as_uint < p, "{}", as_uint);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property test to check that the remainder of a division between 100 pairs of
        // values {x} and {y},generated at random each time tests are run, falls in the
        // range [0, p]. x and y can either take the value of 0 or a large integer.
        // In Cairo, the result of x / y is defined to always satisfy the equation
        // (x / y) * y == x, so the remainder is 0 most of the time.
        fn rem_in_range(x in any::<Felt252>(), y in nonzero_felt252()) {
            let p = &BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();

            let result = x.clone() % y.clone();
            let as_uint = &result.to_biguint();
            prop_assert!(as_uint < p, "{}", as_uint);

            // test reference variant
            let result = x % &y;
            let as_uint = &result.to_biguint();
            prop_assert!(as_uint < p, "{}", as_uint);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property based test that ensures, for 100 Felt252s {x} generated at
        // random each time tests are run, that converting them into the u64 type
        // returns a result that is inside of the range [0, p].
        fn from_u64_and_to_u64_primitive(x in any::<u64>()) {
           let x_felt:Felt252 = Felt252::from_u64(x).unwrap();
           let x_u64:u64 = Felt252::to_u64(&x_felt).unwrap();

            prop_assert_eq!(x, x_u64);
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        fn from_i64_and_to_i64_primitive(x in any::<i64>()) {
            let x = x.checked_abs().unwrap_or(0);
            let x_felt: Felt252 = Felt252::from_i64(x).unwrap();
            let x_i64: i64 = Felt252::to_i64(&x_felt).unwrap();
            prop_assert_eq!(x, x_i64);
        }

        #[test]
        // Property test to check that lcm(x, y) works. Since we're operating in a prime field, lcm
        // will just be the smaller number.
        fn lcm_doesnt_panic(x in any::<Felt252>(), y in any::<Felt252>()) {
            let lcm = x.lcm(&y);
            prop_assert!(lcm == cmp::max(x, y));
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        // Property test to check that is_multiple_of(x, y) works. Since we're operating in a prime field, is_multiple_of
        // will always be true
        fn is_multiple_of_doesnt_panic(x in any::<Felt252>(), y in any::<Felt252>()) {
            prop_assert!(x.is_multiple_of(&y));
        }

        #[test]
        fn divides_doesnt_panic(x in any::<Felt252>(), y in any::<Felt252>()) {
            prop_assert!(x.divides(&y));
        }

        #[test]
        fn gcd_doesnt_panic(x in any::<Felt252>(), y in any::<Felt252>()) {
            let gcd1 = x.gcd(&y);
            let gcd2 = y.gcd(&x);
            prop_assert_eq!(gcd1, gcd2);
        }

        #[test]
        fn is_even(x in any::<Felt252>()) {
            prop_assert_eq!(x.is_even(), x.to_biguint().is_even());
        }

        #[test]
        fn is_odd(x in any::<Felt252>()) {
            prop_assert_eq!(x.is_odd(), x.to_biguint().is_odd());
        }

        /// Tests the additive identity of the implementation of Zero trait for felts
        ///
        /// ```{.text}
        /// x + 0 = x       ∀ x
        /// 0 + x = x       ∀ x
        /// ```
        #[test]
        fn zero_additive_identity(ref x in any::<Felt252>()) {
            let zero = Felt252::zero();
            prop_assert_eq!(x, &(x + &zero));
            prop_assert_eq!(x, &(&zero + x));
        }

        /// Tests the multiplicative identity of the implementation of One trait for felts
        ///
        /// ```{.text}
        /// x * 1 = x       ∀ x
        /// 1 * x = x       ∀ x
        /// ```
        #[test]
        fn one_multiplicative_identity(ref x in any::<Felt252>()) {
            let one = Felt252::one();
            prop_assert_eq!(x, &(x * &one));
            prop_assert_eq!(x, &(&one * x));
        }

        #[test]
        fn felt_is_always_positive(x in any::<Felt252>()) {
            prop_assume!(!x.is_zero());
            prop_assert!(x.is_positive())
        }

        #[test]
        fn felt_is_never_negative(x in any::<Felt252>()) {
            prop_assert!(!x.is_negative())
        }

        #[test]
        fn non_zero_felt_signum_is_always_one(ref x in nonzero_felt252()) {
            let one = Felt252::one();
            prop_assert_eq!(x.signum(), one)
        }

        #[test]
        fn sub_abs(x in any::<Felt252>(), y in any::<Felt252>()) {
            let expected_abs_sub = if x > y {&x - &y} else {&y - &x};

            prop_assert_eq!(x.abs_sub(&y), expected_abs_sub)
        }

        #[test]
        fn abs(x in any::<Felt252>()) {
            prop_assert_eq!(&x, &x.abs())
        }

        #[test]
        fn sqrt_in_range(x in any::<Felt252>()) {
            // we use x = x' * x' so x has a square root
            let x = &x * &x;
            let p = Felt252::prime();

            let sqrt = x.sqrt().to_biguint();
            prop_assert!(sqrt < p, "{}", sqrt);
        }

        #[test]
        fn sqrt_is_inv_square(x in any::<Felt252>()) {
            // we use x = x' * x' so x has a square root
            let x = &x * &x;
            let x_sq = &x * &x;
            let sqrt = x_sq.sqrt();

            if sqrt != x {
                prop_assert_eq!(Felt252::max_value() - sqrt + 1_usize, x);
            } else {
                prop_assert_eq!(sqrt, x);
            }
        }

        #[test]
        fn add_to_u64(x in any::<u64>(), ref felt in any::<Felt252>()) {
            let sum = (felt + x).to_u64();
            prop_assert_eq!(x + felt, sum);
        }

        #[test]
        fn add_to_u64_extremes(x in any::<u64>()) {
            let big_zero = &Felt252::zero();
            let big_max = &Felt252::max_value();
            let big_min = &(big_zero + (i64::MIN as usize));

            let sum_max = (big_max + x).to_u64();
            prop_assert_eq!(x + big_max, sum_max);
            let sum_min = (big_min + x).to_u64();
            prop_assert_eq!(x + big_min, sum_min);
            let sum_zero = (big_zero + x).to_u64();
            prop_assert_eq!(x + big_zero, sum_zero);
        }

        #[test]
        fn add_u32_in_range(x in any::<Felt252>(), y in any::<u32>()) {
            let p = BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();
            let x_add_y = (x + y).to_biguint();
            prop_assert!(x_add_y < p, "{}", x_add_y);
        }

        #[test]
        fn add_u32_is_inv_sub(x in any::<Felt252>(), y in any::<u32>()) {
            let expected_y = (x.clone() + y - x).to_u32().unwrap();
            prop_assert_eq!(expected_y, y, "{}", expected_y);
        }

        #[test]
        fn sub_u32_in_range(x in any::<Felt252>(), y in any::<u32>()) {
            let p = BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();
            let x_sub_y = (x - y).to_biguint();
            prop_assert!(x_sub_y < p, "{}", x_sub_y);
        }

        #[test]
        fn sub_u32_is_inv_add(x in any::<Felt252>(), y in any::<u32>()) {
            prop_assert_eq!(x.clone() - y + y, x)
        }

        #[test]
        fn sub_usize_in_range(x in any::<Felt252>(), y in any::<usize>()) {
            let p = BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();
            let x_sub_y = (x - y).to_biguint();
            prop_assert!(x_sub_y < p, "{}", x_sub_y);
        }

        #[test]
        fn sub_usize_is_inv_add(x in any::<Felt252>(), y in any::<usize>()) {
            prop_assert_eq!(x.clone() - y + y, x)
        }

        #[test]
        fn add_in_range(x in any::<Felt252>(), y in any::<Felt252>()) {
            let p = &BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();

            let sub = x + y;
            let as_uint = &sub.to_biguint();
            prop_assert!(as_uint < p, "{}", as_uint);
        }

        #[test]
        fn add_is_inv_sub(ref x in any::<Felt252>(), ref y in any::<Felt252>()) {
            let expected_y = x + y - x;
            prop_assert_eq!(&expected_y, y, "{}", y);
        }

        #[test]
        fn add_assign_in_range(mut x in any::<Felt252>(), y in any::<Felt252>()) {
            let p = &BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();

            x += y.clone();
            let as_uint = &x.to_biguint();
            prop_assert!(as_uint < p, "{}", as_uint);

            // test reference variant
            x += &y;
            let as_uint = &x.to_biguint();
            prop_assert!(as_uint < p, "{}", as_uint);
        }

        #[test]
        fn felt_to_str(x in any::<Felt252>(), radix in 2_u32..37) {
            let str_x = x.to_str_radix(radix);
            let int_x = x.to_biguint();
            let expected = int_x.to_str_radix(radix);
            prop_assert_eq!(str_x, expected);
        }

        #[test]
        fn bigint_from_felt(x in any::<Felt252>()) {
            prop_assert_eq!(BigInt::from(x.clone()), x.to_bigint());
        }

        #[test]
        fn to_signed_felt_is_negative(x in any::<i128>()) {
            let int = BigInt::from(x);
            let felt = Felt252::from(x);
            prop_assert_eq!(felt.to_signed_felt(), int);
        }
    }

    #[rstest]
    fn add_to_u64_edge_cases(
        #[values(0, 1, u64::MAX)] x: u64,
        #[values(-2, -1, 0, 1, 1i128.neg(), i64::MIN as i128, u64::MAX as i128, u64::MAX as i128 + 1, (u64::MAX as i128).neg())]
        y: i128,
    ) {
        let y = Felt252::from(y);
        assert_eq!(x + &y, (&y + x).to_u64());
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    // Checks that the result of adding two zeroes is zero
    fn sum_zeros_in_range() {
        let x = Felt252::zero();
        let y = Felt252::zero();
        let z = Felt252::zero();
        assert_eq!(x + y, z)
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    // Checks that the result of multiplying two zeroes is zero
    fn mul_zeros_in_range() {
        let x = Felt252::zero();
        let y = Felt252::zero();
        let z = Felt252::zero();
        assert_eq!(x * y, z)
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    // Checks that the result of performing a bit and operation between zeroes is zero
    fn bit_and_zeros_in_range() {
        let x = Felt252::zero();
        let y = Felt252::zero();
        let z = Felt252::zero();
        assert_eq!(&x & &y, z)
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    // Checks that the result of perfforming a bit or operation between zeroes is zero
    fn bit_or_zeros_in_range() {
        let x = Felt252::zero();
        let y = Felt252::zero();
        let z = Felt252::zero();
        assert_eq!(&x | &y, z)
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    // Checks that the result of perfforming a bit xor operation between zeroes is zero
    fn bit_xor_zeros_in_range() {
        let x = Felt252::zero();
        let y = Felt252::zero();
        let z = Felt252::zero();
        assert_eq!(&x ^ &y, z)
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    // Tests that the maximum value a Felt252 can take is equal to (prime - 1)
    fn upper_bound() {
        let prime = &BigUint::parse_bytes(PRIME_STR[2..].as_bytes(), 16).unwrap();
        let unit = BigUint::one();
        let felt_max_value = Felt252::max_value().to_biguint();
        assert_eq!(prime - unit, felt_max_value)
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    // Tests that the minimum value a Felt252 can take is equal to zero.
    fn lower_bound() {
        let zero = BigUint::zero();
        let felt_min_value = Felt252::min_value().to_biguint();
        assert_eq!(zero, felt_min_value)
    }

    #[test]
    fn zero_value() {
        let zero = BigUint::zero();
        let felt_zero = Felt252::zero().to_biguint();
        assert_eq!(zero, felt_zero)
    }

    #[test]
    fn is_zero() {
        let felt_zero = Felt252::zero();
        let felt_non_zero = Felt252::new(3_u32);
        assert!(felt_zero.is_zero());
        assert!(!felt_non_zero.is_zero())
    }

    #[test]
    fn one_value() {
        let one = BigUint::one();
        let felt_one = Felt252::one().to_biguint();
        assert_eq!(one, felt_one)
    }

    #[test]
    fn is_one() {
        let felt_one = Felt252::one();
        let felt_non_one = Felt252::new(8_u32);
        assert!(felt_one.is_one());
        assert!(!felt_non_one.is_one())
    }

    #[test]
    fn signum_of_zero_is_zero() {
        let zero = Felt252::zero();
        assert_eq!(&zero.signum(), &zero)
    }

    #[test]
    fn felt_from_str_radix_failed() {
        let x = Felt252::from_str_radix("abcdefghijk", 16);
        assert!(x.is_err());
        let res = x.unwrap_err().to_string();
        let expected = "ParseFeltError";
        assert_eq!(res, expected)
    }

    #[test]
    fn default_is_zero() {
        assert_eq!(Felt252::default(), Felt252::zero())
    }

    #[test]
    fn from_bytes_ne() {
        let expected = Felt252::zero();
        let bytes = [0; 32];
        let got = Felt252::from_bytes_ne(&bytes);
        assert_eq!(got, expected);
    }

    #[test]
    #[cfg(feature = "parity-scale-codec")]
    // Tests that scale encode operation returns the right value.
    // We chose the value from this example https://docs.substrate.io/reference/scale-codec/.
    fn test_encode_0() {
        let val = Felt252::from_u8(0).unwrap();
        let expected_result = [0];
        assert_eq!(val.encode(), expected_result)
    }

    #[test]
    #[cfg(feature = "parity-scale-codec")]
    // Tests that scale encode operation returns the right value.
    // We chose the value from this example https://docs.substrate.io/reference/scale-codec/.
    fn test_encode_1() {
        let val = Felt252::from_u8(1).unwrap();
        let expected_result = [4];
        assert_eq!(val.encode(), expected_result)
    }

    #[test]
    #[cfg(feature = "parity-scale-codec")]
    // Tests that scale encode operation returns the right value.
    // We chose the value from this example https://docs.substrate.io/reference/scale-codec/.
    fn test_encode_42() {
        let val = Felt252::from_u8(42).unwrap();
        let expected_result = [0xa8];
        assert_eq!(val.encode(), expected_result)
    }

    #[test]
    #[cfg(feature = "parity-scale-codec")]
    // Tests that scale encode operation returns the right value.
    // We chose the value from this example https://docs.substrate.io/reference/scale-codec/.
    fn test_encode_69() {
        let val = Felt252::from_u8(69).unwrap();
        let expected_result = 0x1501u16.to_be_bytes();
        assert_eq!(val.encode(), expected_result)
    }

    #[test]
    #[cfg(feature = "parity-scale-codec")]
    // Tests that scale encode operation returns the right value.
    // We chose the value from this example https://docs.substrate.io/reference/scale-codec/.
    fn test_encode_65535() {
        let val = Felt252::from_u16(65535).unwrap();
        let expected_result = 0xfeff0300u32.to_be_bytes();
        assert_eq!(val.encode(), expected_result)
    }

    #[test]
    #[cfg(feature = "parity-scale-codec")]
    // Tests that scale encode operation returns the right value.
    // We chose the value from this example https://docs.substrate.io/reference/scale-codec/.
    fn test_encode_100000000000000() {
        let val = Felt252::from_u64(100000000000000).unwrap();
        let expected_result = 0x0b00407a10f35au64.to_be_bytes();
        assert_eq!(val.encode(), expected_result[1..])
    }
    #[test]
    #[cfg(feature = "parity-scale-codec")]
    // Tests that scale encode operation returns the right value.
    // We chose the value from this example https://docs.substrate.io/reference/scale-codec/.
    fn test_decode_0() {
        let val = [0];
        let expected_result = Felt252::from_u8(0).unwrap();
        assert_eq!(Felt252::decode(&mut &val[..]).unwrap(), expected_result)
    }

    #[test]
    #[cfg(feature = "parity-scale-codec")]
    // Tests that scale encode operation returns the right value.
    // We chose the value from this example https://docs.substrate.io/reference/scale-codec/.
    fn test_decode_1() {
        let val = [4];
        let expected_result = Felt252::from_u8(1).unwrap();
        assert_eq!(Felt252::decode(&mut &val[..]).unwrap(), expected_result)
    }

    #[test]
    #[cfg(feature = "parity-scale-codec")]
    // Tests that scale encode operation returns the right value.
    // We chose the value from this example https://docs.substrate.io/reference/scale-codec/.
    fn test_decode_42() {
        let val = [0xa8];
        let expected_result = Felt252::from_u8(42).unwrap();
        assert_eq!(Felt252::decode(&mut &val[..]).unwrap(), expected_result)
    }

    #[test]
    #[cfg(feature = "parity-scale-codec")]
    // Tests that scale encode operation returns the right value.
    // We chose the value from this example https://docs.substrate.io/reference/scale-codec/.
    fn test_decode_69() {
        let val = 0x1501u16.to_be_bytes();
        let expected_result = Felt252::from_u8(69).unwrap();
        assert_eq!(Felt252::decode(&mut &val[..]).unwrap(), expected_result)
    }

    #[test]
    #[cfg(feature = "parity-scale-codec")]
    // Tests that scale encode operation returns the right value.
    // We chose the value from this example https://docs.substrate.io/reference/scale-codec/.
    fn test_decode_65535() {
        let val = 0xfeff0300u32.to_be_bytes();
        let expected_result = Felt252::from_u16(65535).unwrap();
        assert_eq!(Felt252::decode(&mut &val[..]).unwrap(), expected_result)
    }

    #[test]
    #[cfg(feature = "parity-scale-codec")]
    // Tests that scale decode operation returns the right value.
    // We chose the value from this example https://docs.substrate.io/reference/scale-codec/.
    fn test_decode_100000000000000() {
        let val = 0x0b00407a10f35au64.to_be_bytes();
        let expected_result = Felt252::from_i64(100000000000000).unwrap();
        assert_eq!(Felt252::decode(&mut &val[1..]).unwrap(), expected_result)
    }
}
