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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseFeltError;

impl fmt::Display for ParseFeltError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{ParseFeltError:?}")
    }
}
