use crate::stdlib::{borrow::Cow, collections::HashMap, fmt, prelude::*};

use crate::{
    types::relocatable::{MaybeRelocatable, Relocatable},
    utils::from_relocatable_to_indexes,
    vm::errors::memory_errors::MemoryError,
};
//use bitvec::prelude as bv;
use core::cmp::Ordering;
use felt::Felt252;
use num_traits::ToPrimitive;

pub struct ValidationRule(
    #[allow(clippy::type_complexity)]
    pub  Box<dyn Fn(&Memory, Relocatable) -> Result<(usize, usize), MemoryError>>,
);

#[derive(Clone, Eq, Ord, PartialEq, PartialOrd, Debug)]
pub(crate) struct MemoryCell(MaybeRelocatable, bool);

impl MemoryCell {
    pub fn new(value: MaybeRelocatable) -> Self {
        MemoryCell(value, false)
    }

    pub fn mark_accessed(&mut self) {
        self.1 = true
    }

    pub fn is_accessed(&self) -> bool {
        self.1
    }

    pub fn get_value(&self) -> &MaybeRelocatable {
        &self.0
    }

    pub fn get_value_mut(&mut self) -> &mut MaybeRelocatable {
        &mut self.0
    }
}

// TODO: smallvec
pub struct AddressSet(Vec<(usize, usize)>);

// PERFORMANCE: we assume there's a low number of ranges in the address set,
// so we use a simple linear search. Also assume we're not marking the same
// offset twice.
impl AddressSet {
    pub(crate) fn new() -> Self {
        Self(Vec::new())
    }

    #[inline(always)]
    pub(crate) fn contains(&self, offset: usize) -> bool {
        self.0.binary_search_by(|range| {
            if offset < range.0 {
                Ordering::Greater
            } else if offset > range.1 {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        })
        .is_ok()
    }

    #[inline(always)]
    pub(crate) fn extend(&mut self, (start, end): (usize, usize)) {
        let pp = self.0.partition_point(|range| range.1 < start);
        if pp == self.0.len() {
            self.0.push((start, end));
            return;
        }
        if pp == 0 {
            if self.0[0].0 == end + 1 {
                self.0[0].0 = start;
                return;
            }
            self.0.insert(0, (start, end));
            return;
        }
        if self.0[pp - 1].1 == start - 1 {
            if self.0[pp].0 == end + 1 {
                self.0[pp - 1].1 = self.0[pp].1;
                self.0.remove(pp);
                return;
            }
            self.0[pp - 1].1 = end;
        }
        if self.0[pp].0 == end + 1 {
            self.0[pp].0 = start;
            return;
        }
    }
}

/*
#[cfg(test)]
impl AddressSet {
    pub(crate) fn len(&self) -> usize {
        self.0.iter().map(|bit| *bit as usize).sum::<usize>()
    }
}
*/

pub struct Memory {
    pub(crate) data: Vec<Vec<Option<MemoryCell>>>,
    pub(crate) temp_data: Vec<Vec<Option<MemoryCell>>>,
    // relocation_rules's keys map to temp_data's indices and therefore begin at
    // zero; that is, segment_index = -1 maps to key 0, -2 to key 1...
    pub(crate) relocation_rules: HashMap<usize, Relocatable>,
    validation_rules: Vec<Option<(ValidationRule, AddressSet)>>,
}

impl Memory {
    pub fn new() -> Memory {
        Memory {
            data: Vec::<Vec<Option<MemoryCell>>>::new(),
            temp_data: Vec::<Vec<Option<MemoryCell>>>::new(),
            relocation_rules: HashMap::new(),
            validation_rules: Vec::with_capacity(20),
        }
    }

    /// Inserts a value into a memory address
    /// Will return an Error if the segment index given by the address corresponds to a non-allocated segment,
    /// or if the inserted value is inconsistent with the current value at the memory cell
    /// If the address isnt contiguous with previously inserted data, memory gaps will be represented by None values
    pub fn insert<V>(&mut self, key: Relocatable, val: V) -> Result<(), MemoryError>
    where
        MaybeRelocatable: From<V>,
    {
        let val = MaybeRelocatable::from(val);
        let (value_index, value_offset) = from_relocatable_to_indexes(key);

        let data = if key.segment_index.is_negative() {
            &mut self.temp_data
        } else {
            &mut self.data
        };

        let data_len = data.len();
        let segment = data
            .get_mut(value_index)
            .ok_or_else(|| MemoryError::UnallocatedSegment(Box::new((value_index, data_len))))?;

        //Check if the element is inserted next to the last one on the segment
        //Forgoing this check would allow data to be inserted in a different index
        let (len, capacity) = (segment.len(), segment.capacity());
        if len <= value_offset {
            let new_len = value_offset
                .checked_add(1)
                .ok_or(MemoryError::VecCapacityExceeded)?;
            segment
                .try_reserve(new_len.saturating_sub(capacity))
                .map_err(|_| MemoryError::VecCapacityExceeded)?;
            segment.resize(new_len, None);
        }
        // At this point there's *something* in there

        match segment[value_offset] {
            None => segment[value_offset] = Some(MemoryCell::new(val)),
            Some(ref current_cell) => {
                if current_cell.get_value() != &val {
                    //Existing memory cannot be changed
                    return Err(MemoryError::InconsistentMemory(Box::new((
                        key,
                        current_cell.get_value().clone(),
                        val,
                    ))));
                }
            }
        };
        self.validate_memory_cell(key)
    }

    /// Retrieve a value from memory (either normal or temporary) and apply relocation rules
    pub(crate) fn get<'a, 'b: 'a, K: 'a>(&'b self, key: &'a K) -> Option<Cow<MaybeRelocatable>>
    where
        Relocatable: TryFrom<&'a K>,
    {
        let relocatable: Relocatable = key.try_into().ok()?;

        let data = if relocatable.segment_index.is_negative() {
            &self.temp_data
        } else {
            &self.data
        };
        let (i, j) = from_relocatable_to_indexes(relocatable);
        Some(self.relocate_value(data.get(i)?.get(j)?.as_ref()?.get_value()))
    }

    // Version of Memory.relocate_value() that doesn't require a self reference
    fn relocate_address(
        addr: Relocatable,
        relocation_rules: &HashMap<usize, Relocatable>,
    ) -> MaybeRelocatable {
        let segment_idx = addr.segment_index;
        if segment_idx >= 0 {
            return addr.into();
        }

        // Adjust the segment index to begin at zero, as per the struct field's
        match relocation_rules.get(&(-(segment_idx + 1) as usize)) {
            Some(x) => (x + addr.offset).into(),
            None => addr.into(),
        }
    }

    /// Relocates the memory according to the relocation rules and clears `self.relocaction_rules`.
    pub fn relocate_memory(&mut self) -> Result<(), MemoryError> {
        if self.relocation_rules.is_empty() || self.temp_data.is_empty() {
            return Ok(());
        }
        // Relocate temporary addresses in memory
        for segment in self.data.iter_mut().chain(self.temp_data.iter_mut()) {
            for cell in segment.iter_mut().flatten() {
                let value = cell.get_value_mut();
                match value {
                    MaybeRelocatable::RelocatableValue(addr) if addr.segment_index < 0 => {
                        *value = Memory::relocate_address(*addr, &self.relocation_rules);
                    }
                    _ => {}
                }
            }
        }
        // Move relocated temporary memory into the real memory
        for index in (0..self.temp_data.len()).rev() {
            if let Some(base_addr) = self.relocation_rules.get(&index) {
                let data_segment = self.temp_data.remove(index);
                // Insert the to-be relocated segment into the real memory
                let mut addr = *base_addr;
                if let Some(s) = self.data.get_mut(addr.segment_index as usize) {
                    s.reserve_exact(data_segment.len())
                }
                for cell in data_segment {
                    if let Some(cell) = cell {
                        // Rely on Memory::insert to catch memory inconsistencies
                        self.insert(addr, cell.get_value())?;
                    }
                    addr = (addr + 1)?;
                }
            }
        }
        self.relocation_rules.clear();
        Ok(())
    }

    /// Add a new relocation rule.
    ///
    /// Will return an error if any of the following conditions are not met:
    ///   - Source address's segment must be negative (temporary).
    ///   - Source address's offset must be zero.
    ///   - There shouldn't already be relocation at the source segment.
    pub(crate) fn add_relocation_rule(
        &mut self,
        src_ptr: Relocatable,
        dst_ptr: Relocatable,
    ) -> Result<(), MemoryError> {
        if src_ptr.segment_index >= 0 {
            return Err(MemoryError::AddressNotInTemporarySegment(
                src_ptr.segment_index,
            ));
        }
        if src_ptr.offset != 0 {
            return Err(MemoryError::NonZeroOffset(src_ptr.offset));
        }

        // Adjust the segment index to begin at zero, as per the struct field's
        // comment.
        let segment_index = -(src_ptr.segment_index + 1) as usize;
        if self.relocation_rules.contains_key(&segment_index) {
            return Err(MemoryError::DuplicatedRelocation(src_ptr.segment_index));
        }

        self.relocation_rules.insert(segment_index, dst_ptr);
        Ok(())
    }

    /// Gets the value from memory address as a Felt252 value.
    /// Returns an Error if the value at the memory address is missing or not a Felt252.
    pub fn get_integer(&self, key: Relocatable) -> Result<Cow<Felt252>, MemoryError> {
        match self
            .get(&key)
            .ok_or_else(|| MemoryError::UnknownMemoryCell(Box::new(key)))?
        {
            Cow::Borrowed(MaybeRelocatable::Int(int)) => Ok(Cow::Borrowed(int)),
            Cow::Owned(MaybeRelocatable::Int(int)) => Ok(Cow::Owned(int)),
            _ => Err(MemoryError::ExpectedInteger(Box::new(key))),
        }
    }

    /// Gets the value from memory address as a Relocatable value.
    /// Returns an Error if the value at the memory address is missing or not a Relocatable.
    pub fn get_relocatable(&self, key: Relocatable) -> Result<Relocatable, MemoryError> {
        match self
            .get(&key)
            .ok_or_else(|| MemoryError::UnknownMemoryCell(Box::new(key)))?
        {
            Cow::Borrowed(MaybeRelocatable::RelocatableValue(rel)) => Ok(*rel),
            Cow::Owned(MaybeRelocatable::RelocatableValue(rel)) => Ok(rel),
            _ => Err(MemoryError::ExpectedRelocatable(Box::new(key))),
        }
    }

    /// Inserts a value into memory
    /// Returns an error if the memory cell asignment is invalid
    pub fn insert_value<T: Into<MaybeRelocatable>>(
        &mut self,
        key: Relocatable,
        val: T,
    ) -> Result<(), MemoryError> {
        self.insert(key, &val.into())
    }

    pub fn add_validation_rule(&mut self, segment_index: usize, rule: ValidationRule) {
        let new_len = self.validation_rules.len().max(segment_index + 1);
        self.validation_rules.resize_with(new_len, || None);
        self.validation_rules.insert(segment_index, Some((rule, AddressSet::new())));
    }

    fn validate_memory_cell(&mut self, addr: Relocatable) -> Result<(), MemoryError> {
        let mut res = Ok(());
        let mut rules_tmp = Vec::new();
        core::mem::swap(&mut self.validation_rules, &mut rules_tmp);
        if let Some(Some((rule, validated_addresses))) = addr
            .segment_index
            .to_usize()
            .and_then(|x| rules_tmp.get_mut(x))
        {
            if validated_addresses.contains(addr.offset) {
                return Ok(());
            }
            let validated = rule.0(self, addr);
            // NOTE: we can't simply `?` here because we need to restore the validation rules at the end
            match validated {
                Ok(addresses) => {
                    validated_addresses.extend(addresses);
                },
                Err(e) => {
                    res = Err(e);
                },
            }
        }
        self.validation_rules = rules_tmp;
        res
    }

    ///Applies validation_rules to the current memory
    pub fn validate_existing_memory(&mut self) -> Result<(), MemoryError> {
        let mut res = Ok(());
        let mut rules_tmp = Vec::new();
        core::mem::swap(&mut self.validation_rules, &mut rules_tmp);
        for (segment, rule) in rules_tmp.iter_mut().enumerate() {
            let Some((rule, ref mut validated_addresses)) = rule else {
                continue;
            };
            let Some(len) = self.data.get(segment).map(|s| s.len()) else {
                continue;
            };
            for offset in 0..len {
                let addr = (segment as isize, offset).into();
                if validated_addresses.contains(offset) {
                    continue;
                }
                let validated = rule.0(self, addr);
                // NOTE: we can't simply `?` here because we need to restore the validation rules at the end
                match validated {
                    Ok(addresses) => {
                        validated_addresses.extend(addresses);
                    },
                    // NOTE: we need to break here, otherwise we lose the rules
                    Err(e) => {
                        res = Err(e);
                        break;
                    },
                }
            }
        }
        self.validation_rules = rules_tmp;
        res
    }

    /// Compares two ranges of values in memory of length `len`
    /// Returns the ordering and the first relative position at which they differ
    /// Special cases:
    /// - `lhs` exists in memory but `rhs` doesn't -> (Ordering::Greater, 0)
    /// - `rhs` exists in memory but `lhs` doesn't -> (Ordering::Less, 0)
    /// - None of `lhs` or `rhs` exist in memory -> (Ordering::Equal, 0)
    /// Everything else behaves much like `memcmp` in C.
    /// This is meant as an optimization for hints to avoid allocations.
    pub(crate) fn memcmp(
        &self,
        lhs: Relocatable,
        rhs: Relocatable,
        len: usize,
    ) -> (Ordering, usize) {
        let get_segment = |idx: isize| {
            if idx.is_negative() {
                self.temp_data.get(-(idx + 1) as usize)
            } else {
                self.data.get(idx as usize)
            }
        };
        match (
            get_segment(lhs.segment_index),
            get_segment(rhs.segment_index),
        ) {
            (None, None) => {
                return (Ordering::Equal, 0);
            }
            (Some(_), None) => {
                return (Ordering::Greater, 0);
            }
            (None, Some(_)) => {
                return (Ordering::Less, 0);
            }
            (Some(lhs_segment), Some(rhs_segment)) => {
                let (lhs_start, rhs_start) = (lhs.offset, rhs.offset);
                for i in 0..len {
                    let (lhs, rhs) = (
                        lhs_segment.get(lhs_start + i),
                        rhs_segment.get(rhs_start + i),
                    );
                    let ord = lhs.cmp(&rhs);
                    if ord == Ordering::Equal {
                        continue;
                    }
                    return (ord, i);
                }
            }
        };
        (Ordering::Equal, len)
    }

    /// Compares two ranges of values in memory of length `len`
    /// Returns the ordering and the first relative position at which they differ
    /// Special cases:
    /// - `lhs` exists in memory but `rhs` doesn't -> (Ordering::Greater, 0)
    /// - `rhs` exists in memory but `lhs` doesn't -> (Ordering::Less, 0)
    /// - None of `lhs` or `rhs` exist in memory -> (Ordering::Equal, 0)
    /// Everything else behaves much like `memcmp` in C.
    /// This is meant as an optimization for hints to avoid allocations.
    pub(crate) fn mem_eq(&self, lhs: Relocatable, rhs: Relocatable, len: usize) -> bool {
        if lhs == rhs {
            return true;
        }
        let get_segment = |idx: isize| {
            if idx.is_negative() {
                self.temp_data.get(-(idx + 1) as usize)
            } else {
                self.data.get(idx as usize)
            }
        };
        match (
            get_segment(lhs.segment_index).and_then(|s| s.get(lhs.offset..)),
            get_segment(rhs.segment_index).and_then(|s| s.get(rhs.offset..)),
        ) {
            (Some(lhs), Some(rhs)) => {
                let (lhs_len, rhs_len) = (lhs.len().min(len), rhs.len().min(len));
                if lhs_len != rhs_len {
                    return false;
                }
                lhs[..lhs_len] == rhs[..rhs_len]
            }
            (None, None) => true,
            _ => false,
        }
    }

    /// Gets a range of memory values from addr to addr + size
    /// The outputed range may contain gaps if the original memory has them
    pub fn get_range(&self, addr: Relocatable, size: usize) -> Vec<Option<Cow<MaybeRelocatable>>> {
        let mut values = Vec::new();

        for i in 0..size {
            values.push((addr + i).ok().and_then(|x| self.get(&x)));
        }

        values
    }

    /// Gets a range of memory values from addr to addr + size
    /// Fails if there if any of the values inside the range is missing (memory gap)
    pub fn get_continuous_range(
        &self,
        addr: Relocatable,
        size: usize,
    ) -> Result<Vec<MaybeRelocatable>, MemoryError> {
        let mut values = Vec::with_capacity(size);

        for i in 0..size {
            values.push(match self.get(&(addr + i)?) {
                Some(elem) => elem.into_owned(),
                None => return Err(MemoryError::GetRangeMemoryGap(Box::new((addr, size)))),
            });
        }

        Ok(values)
    }

    /// Gets a range of Felt252 memory values from addr to addr + size
    /// Fails if there if any of the values inside the range is missing (memory gap),
    /// or is not a Felt252
    pub fn get_integer_range(
        &self,
        addr: Relocatable,
        size: usize,
    ) -> Result<Vec<Cow<Felt252>>, MemoryError> {
        let mut values = Vec::new();

        for i in 0..size {
            values.push(self.get_integer((addr + i)?)?);
        }

        Ok(values)
    }

    pub fn mark_as_accessed(&mut self, addr: Relocatable) {
        let (i, j) = from_relocatable_to_indexes(addr);
        let data = if addr.segment_index < 0 {
            &mut self.temp_data
        } else {
            &mut self.data
        };
        let cell = data.get_mut(i).and_then(|x| x.get_mut(j));
        if let Some(Some(cell)) = cell {
            cell.mark_accessed()
        }
    }

    pub fn get_amount_of_accessed_addresses_for_segment(
        &self,
        segment_index: usize,
    ) -> Option<usize> {
        let segment = self.data.get(segment_index)?;
        Some(
            segment
                .iter()
                .filter(|x| {
                    if let Some(cell) = x {
                        cell.is_accessed()
                    } else {
                        false
                    }
                })
                .count(),
        )
    }
}

impl fmt::Display for Memory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, segment) in self.temp_data.iter().enumerate() {
            for (j, cell) in segment.iter().enumerate() {
                if let Some(cell) = cell {
                    let temp_segment = i + 1;
                    let elem = cell.get_value();
                    writeln!(f, "(-{temp_segment},{j}) : {elem}")?;
                }
            }
        }
        for (i, segment) in self.data.iter().enumerate() {
            for (j, cell) in segment.iter().enumerate() {
                if let Some(cell) = cell {
                    let elem = cell.get_value();
                    writeln!(f, "({i},{j}) : {elem}")?;
                }
            }
        }
        writeln!(f, "}}")
    }
}

/// Applies `relocation_rules` to a value
pub(crate) trait RelocateValue<'a, Input: 'a, Output: 'a> {
    fn relocate_value(&self, value: Input) -> Output;
}

impl RelocateValue<'_, Relocatable, Relocatable> for Memory {
    fn relocate_value(&self, addr: Relocatable) -> Relocatable {
        let segment_idx = addr.segment_index;
        if segment_idx >= 0 {
            return addr;
        }

        // Adjust the segment index to begin at zero, as per the struct field's
        // comment.
        match self.relocation_rules.get(&(-(segment_idx + 1) as usize)) {
            Some(x) => x + addr.offset,
            None => addr,
        }
    }
}

impl<'a> RelocateValue<'a, &'a Felt252, &'a Felt252> for Memory {
    fn relocate_value(&self, value: &'a Felt252) -> &'a Felt252 {
        value
    }
}

impl<'a> RelocateValue<'a, &'a MaybeRelocatable, Cow<'a, MaybeRelocatable>> for Memory {
    fn relocate_value(&self, value: &'a MaybeRelocatable) -> Cow<'a, MaybeRelocatable> {
        match value {
            MaybeRelocatable::Int(_) => Cow::Borrowed(value),
            MaybeRelocatable::RelocatableValue(addr) => {
                Cow::Owned(self.relocate_value(*addr).into())
            }
        }
    }
}

impl Default for Memory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod memory_tests {
    use core::ops::Shl;

    use super::*;
    use crate::{
        relocatable,
        types::instance_definitions::ecdsa_instance_def::EcdsaInstanceDef,
        utils::test_utils::*,
        vm::{
            runners::builtin_runner::{RangeCheckBuiltinRunner, SignatureBuiltinRunner},
            vm_memory::memory_segments::MemorySegmentManager,
        },
    };
    use assert_matches::assert_matches;
    use felt::felt_str;
    use num_traits::One;

    use crate::vm::errors::memory_errors::MemoryError;

    use crate::utils::test_utils::memory_from_memory;
    use crate::utils::test_utils::memory_inner;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn insert_and_get_succesful() {
        let key = Relocatable::from((0, 0));
        let val = MaybeRelocatable::from(Felt252::new(5));
        let mut memory = Memory::new();
        memory.data.push(Vec::new());
        memory.insert(key, &val).unwrap();
        assert_eq!(
            memory.get(&key).unwrap().as_ref(),
            &MaybeRelocatable::from(Felt252::new(5))
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn get_valuef_from_temp_segment() {
        let mut memory = Memory::new();
        memory.temp_data = vec![vec![
            None,
            None,
            Some(MemoryCell::new(mayberelocatable!(8))),
        ]];
        assert_eq!(
            memory.get(&mayberelocatable!(-1, 2)).unwrap().as_ref(),
            &mayberelocatable!(8),
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn insert_value_in_temp_segment() {
        let key = Relocatable::from((-1, 3));
        let val = MaybeRelocatable::from(Felt252::new(8));
        let mut memory = Memory::new();
        memory.temp_data.push(Vec::new());
        memory.insert(key, &val).unwrap();
        assert_eq!(
            memory.temp_data[0][3],
            Some(MemoryCell::new(MaybeRelocatable::from(Felt252::new(8))))
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn insert_and_get_from_temp_segment_succesful() {
        let key = Relocatable::from((-1, 0));
        let val = MaybeRelocatable::from(Felt252::new(5));
        let mut memory = Memory::new();
        memory.temp_data.push(Vec::new());
        memory.insert(key, &val).unwrap();
        assert_eq!(
            memory.get(&key).unwrap().as_ref(),
            &MaybeRelocatable::from(Felt252::new(5)),
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn insert_and_get_from_temp_segment_failed() {
        let key = relocatable!(-1, 1);
        let mut memory = Memory::new();
        memory.temp_data = vec![vec![None, Some(MemoryCell::new(mayberelocatable!(8)))]];
        assert_eq!(
            memory.insert(key, &mayberelocatable!(5)),
            Err(MemoryError::InconsistentMemory(Box::new((
                relocatable!(-1, 1),
                mayberelocatable!(8),
                mayberelocatable!(5)
            ))))
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn get_non_allocated_memory() {
        let key = Relocatable::from((0, 0));
        let memory = Memory::new();
        assert_eq!(memory.get(&key), None);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn get_non_existant_element() {
        let key = Relocatable::from((0, 0));
        let memory = Memory::new();
        assert_eq!(memory.get(&key), None);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn insert_non_allocated_memory() {
        let key = Relocatable::from((0, 0));
        let val = MaybeRelocatable::from(Felt252::new(5));
        let mut memory = Memory::new();
        let error = memory.insert(key, &val);
        assert_eq!(
            error,
            Err(MemoryError::UnallocatedSegment(Box::new((0, 0))))
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn insert_inconsistent_memory() {
        let key = Relocatable::from((0, 0));
        let val_a = MaybeRelocatable::from(Felt252::new(5));
        let val_b = MaybeRelocatable::from(Felt252::new(6));
        let mut memory = Memory::new();
        memory.data.push(Vec::new());
        memory
            .insert(key, &val_a)
            .expect("Unexpected memory insert fail");
        let error = memory.insert(key, &val_b);
        assert_eq!(
            error,
            Err(MemoryError::InconsistentMemory(Box::new((
                key, val_a, val_b
            ))))
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn insert_non_contiguous_element() {
        let key_a = Relocatable::from((0, 0));
        let key_b = Relocatable::from((0, 2));
        let val = MaybeRelocatable::from(Felt252::new(5));
        let mut memory = Memory::new();
        memory.data.push(Vec::new());
        memory.insert(key_a, &val).unwrap();
        memory.insert(key_b, &val).unwrap();
        assert_eq!(memory.get(&key_b).unwrap().as_ref(), &val);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn insert_non_contiguous_element_memory_gaps_none() {
        let key_a = Relocatable::from((0, 0));
        let key_b = Relocatable::from((0, 5));
        let val = MaybeRelocatable::from(Felt252::new(5));
        let mut memory = Memory::new();
        memory.data.push(Vec::new());
        memory.insert(key_a, &val).unwrap();
        memory.insert(key_b, &val).unwrap();
        assert_eq!(memory.get(&key_b).unwrap().as_ref(), &val);
        assert_eq!(memory.get(&MaybeRelocatable::from((0, 1))), None);
        assert_eq!(memory.get(&MaybeRelocatable::from((0, 2))), None);
        assert_eq!(memory.get(&MaybeRelocatable::from((0, 3))), None);
        assert_eq!(memory.get(&MaybeRelocatable::from((0, 4))), None);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn validate_existing_memory_for_range_check_within_bounds() {
        let mut builtin = RangeCheckBuiltinRunner::new(Some(8), 8, true);
        let mut segments = MemorySegmentManager::new();
        builtin.initialize_segments(&mut segments);
        builtin.add_validation_rule(&mut segments.memory);
        for _ in 0..3 {
            segments.add();
        }

        segments
            .memory
            .insert(
                Relocatable::from((0, 0)),
                &MaybeRelocatable::from(Felt252::new(45)),
            )
            .unwrap();
        segments.memory.validate_existing_memory().unwrap();
        assert!(segments
            .memory
            .validation_rules[0].as_ref().unwrap().1
            .contains(0));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn validate_existing_memory_for_range_check_outside_bounds() {
        let mut builtin = RangeCheckBuiltinRunner::new(Some(8), 8, true);
        let mut segments = MemorySegmentManager::new();
        segments.add();
        builtin.initialize_segments(&mut segments);
        segments
            .memory
            .insert(
                Relocatable::from((1, 0)),
                &MaybeRelocatable::from(Felt252::new(-10)),
            )
            .unwrap();
        builtin.add_validation_rule(&mut segments.memory);
        let error = segments.memory.validate_existing_memory();
        assert_eq!(
            error,
            Err(MemoryError::RangeCheckNumOutOfBounds(Box::new((
                Felt252::new(-10),
                Felt252::one().shl(128_u32)
            ))))
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn validate_existing_memory_for_invalid_signature() {
        let mut builtin = SignatureBuiltinRunner::new(&EcdsaInstanceDef::default(), true);
        let mut segments = MemorySegmentManager::new();
        builtin.initialize_segments(&mut segments);
        segments.memory = memory![
            (
                (0, 0),
                (
                    "874739451078007766457464989774322083649278607533249481151382481072868806602",
                    10
                )
            ),
            (
                (0, 1),
                (
                    "-1472574760335685482768423018116732869320670550222259018541069375211356613248",
                    10
                )
            )
        ];
        builtin.add_validation_rule(&mut segments.memory);
        let error = segments.memory.validate_existing_memory();
        assert_eq!(
            error,
            Err(MemoryError::SignatureNotFound(Box::new((0, 0).into())))
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn validate_existing_memory_for_valid_signature() {
        let mut builtin = SignatureBuiltinRunner::new(&EcdsaInstanceDef::default(), true);

        let signature_r = felt_str!(
            "1839793652349538280924927302501143912227271479439798783640887258675143576352"
        );
        let signature_s = felt_str!(
            "1819432147005223164874083361865404672584671743718628757598322238853218813979"
        );

        builtin
            .add_signature(Relocatable::from((1, 0)), &(signature_r, signature_s))
            .unwrap();

        let mut segments = MemorySegmentManager::new();

        segments.memory = memory![
            (
                (1, 0),
                (
                    "874739451078007766457464989774322083649278607533249481151382481072868806602",
                    10
                )
            ),
            ((1, 1), 2)
        ];

        builtin.initialize_segments(&mut segments);

        builtin.add_validation_rule(&mut segments.memory);

        let result = segments.memory.validate_existing_memory();

        assert_eq!(result, Ok(()))
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn validate_existing_memory_for_range_check_relocatable_value() {
        let mut builtin = RangeCheckBuiltinRunner::new(Some(8), 8, true);
        let mut segments = MemorySegmentManager::new();
        builtin.initialize_segments(&mut segments);
        segments.memory = memory![((0, 0), (0, 4))];
        builtin.add_validation_rule(&mut segments.memory);
        let error = segments.memory.validate_existing_memory();
        assert_eq!(
            error,
            Err(MemoryError::RangeCheckFoundNonInt(Box::new(relocatable!(
                0, 0
            ))))
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn validate_existing_memory_for_range_check_out_of_bounds_diff_segment() {
        let mut builtin = RangeCheckBuiltinRunner::new(Some(8), 8, true);
        let mut segments = MemorySegmentManager::new();
        segments.memory = Memory::new();
        segments.add();
        builtin.initialize_segments(&mut segments);
        segments
            .memory
            .insert(
                Relocatable::from((0, 0)),
                &MaybeRelocatable::from(Felt252::new(-45)),
            )
            .unwrap();
        builtin.add_validation_rule(&mut segments.memory);
        assert_eq!(segments.memory.validate_existing_memory(), Ok(()));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn get_integer_valid() {
        let memory = memory![((0, 0), 10)];
        assert_eq!(
            memory
                .get_integer(Relocatable::from((0, 0)))
                .unwrap()
                .as_ref(),
            &Felt252::new(10)
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn get_integer_invalid_expected_integer() {
        let mut segments = MemorySegmentManager::new();
        segments.add();
        segments
            .memory
            .insert(Relocatable::from((0, 0)), &MaybeRelocatable::from((0, 10)))
            .unwrap();
        assert_matches!(
            segments.memory.get_integer(Relocatable::from((0, 0))),
            Err(MemoryError::ExpectedInteger(
                bx
            )) if *bx == Relocatable::from((0, 0))
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn default_memory() {
        let mem: Memory = Default::default();
        assert_eq!(mem.data.len(), 0);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn insert_and_get_temporary_succesful() {
        let mut memory = Memory::new();
        memory.temp_data.push(Vec::new());

        let key = Relocatable::from((-1, 0));
        let val = MaybeRelocatable::from(Felt252::new(5));
        memory.insert(key, &val).unwrap();

        assert_eq!(memory.get(&key).unwrap().as_ref(), &val);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn add_relocation_rule() {
        let mut memory = Memory::new();

        assert_eq!(
            memory.add_relocation_rule((-1, 0).into(), (1, 2).into()),
            Ok(()),
        );
        assert_eq!(
            memory.add_relocation_rule((-2, 0).into(), (-1, 1).into()),
            Ok(()),
        );
        assert_eq!(
            memory.add_relocation_rule((5, 0).into(), (0, 0).into()),
            Err(MemoryError::AddressNotInTemporarySegment(5)),
        );
        assert_eq!(
            memory.add_relocation_rule((-3, 6).into(), (0, 0).into()),
            Err(MemoryError::NonZeroOffset(6)),
        );
        assert_eq!(
            memory.add_relocation_rule((-1, 0).into(), (0, 0).into()),
            Err(MemoryError::DuplicatedRelocation(-1)),
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn relocate_value_bigint() {
        let mut memory = Memory::new();
        memory
            .add_relocation_rule((-1, 0).into(), (2, 0).into())
            .unwrap();
        memory
            .add_relocation_rule((-2, 0).into(), (2, 2).into())
            .unwrap();

        // Test when value is Some(BigInt):
        assert_eq!(
            memory.relocate_value(&MaybeRelocatable::Int(Felt252::new(0))),
            Cow::Owned(MaybeRelocatable::Int(Felt252::new(0))),
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn relocate_value_mayberelocatable() {
        let mut memory = Memory::new();
        memory
            .add_relocation_rule((-1, 0).into(), (2, 0).into())
            .unwrap();
        memory
            .add_relocation_rule((-2, 0).into(), (2, 2).into())
            .unwrap();

        // Test when value is Some(MaybeRelocatable) with segment_index >= 0:
        assert_eq!(
            memory.relocate_value(&MaybeRelocatable::RelocatableValue((0, 0).into())),
            Cow::Owned(MaybeRelocatable::RelocatableValue((0, 0).into())),
        );
        assert_eq!(
            memory.relocate_value(&MaybeRelocatable::RelocatableValue((5, 0).into())),
            Cow::Owned(MaybeRelocatable::RelocatableValue((5, 0).into())),
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn relocate_value_mayberelocatable_temporary_segment_no_rules() {
        let mut memory = Memory::new();
        memory
            .add_relocation_rule((-1, 0).into(), (2, 0).into())
            .unwrap();
        memory
            .add_relocation_rule((-2, 0).into(), (2, 2).into())
            .unwrap();

        // Test when value is Some(MaybeRelocatable) with segment_index < 0 and
        // there are no applicable relocation rules:
        assert_eq!(
            memory.relocate_value(&MaybeRelocatable::RelocatableValue((-5, 0).into())),
            Cow::Owned(MaybeRelocatable::RelocatableValue((-5, 0).into())),
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn relocate_value_mayberelocatable_temporary_segment_rules() {
        let mut memory = Memory::new();
        memory
            .add_relocation_rule((-1, 0).into(), (2, 0).into())
            .unwrap();
        memory
            .add_relocation_rule((-2, 0).into(), (2, 2).into())
            .unwrap();

        // Test when value is Some(MaybeRelocatable) with segment_index < 0 and
        // there are applicable relocation rules:
        assert_eq!(
            memory.relocate_value(&MaybeRelocatable::RelocatableValue((-1, 0).into())),
            Cow::Owned(MaybeRelocatable::RelocatableValue((2, 0).into())),
        );
        assert_eq!(
            memory.relocate_value(&MaybeRelocatable::RelocatableValue((-2, 0).into())),
            Cow::Owned(MaybeRelocatable::RelocatableValue((2, 2).into())),
        );
        assert_eq!(
            memory.relocate_value(&MaybeRelocatable::RelocatableValue((-1, 5).into())),
            Cow::Owned(MaybeRelocatable::RelocatableValue((2, 5).into())),
        );
        assert_eq!(
            memory.relocate_value(&MaybeRelocatable::RelocatableValue((-2, 5).into())),
            Cow::Owned(MaybeRelocatable::RelocatableValue((2, 7).into())),
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn get_range_for_continuous_memory() {
        let memory = memory![((1, 0), 2), ((1, 1), 3), ((1, 2), 4)];

        let value1 = MaybeRelocatable::from(Felt252::new(2));
        let value2 = MaybeRelocatable::from(Felt252::new(3));
        let value3 = MaybeRelocatable::from(Felt252::new(4));

        let expected_vec = vec![
            Some(Cow::Borrowed(&value1)),
            Some(Cow::Borrowed(&value2)),
            Some(Cow::Borrowed(&value3)),
        ];
        assert_eq!(memory.get_range(Relocatable::from((1, 0)), 3), expected_vec);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn get_range_for_non_continuous_memory() {
        let memory = memory![((1, 0), 2), ((1, 1), 3), ((1, 3), 4)];

        let value1 = MaybeRelocatable::from(Felt252::new(2));
        let value2 = MaybeRelocatable::from(Felt252::new(3));
        let value3 = MaybeRelocatable::from(Felt252::new(4));

        let expected_vec = vec![
            Some(Cow::Borrowed(&value1)),
            Some(Cow::Borrowed(&value2)),
            None,
            Some(Cow::Borrowed(&value3)),
        ];
        assert_eq!(memory.get_range(Relocatable::from((1, 0)), 4), expected_vec);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn get_continuous_range_for_continuous_memory() {
        let memory = memory![((1, 0), 2), ((1, 1), 3), ((1, 2), 4)];

        let value1 = MaybeRelocatable::from(Felt252::new(2));
        let value2 = MaybeRelocatable::from(Felt252::new(3));
        let value3 = MaybeRelocatable::from(Felt252::new(4));

        let expected_vec = vec![value1, value2, value3];
        assert_eq!(
            memory.get_continuous_range(Relocatable::from((1, 0)), 3),
            Ok(expected_vec)
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn get_continuous_range_for_non_continuous_memory() {
        let memory = memory![((1, 0), 2), ((1, 1), 3), ((1, 3), 4)];

        assert_eq!(
            memory.get_continuous_range(Relocatable::from((1, 0)), 3),
            Err(MemoryError::GetRangeMemoryGap(Box::new(((1, 0).into(), 3))))
        );
    }

    /// Test that relocate_memory() works when there are no relocation rules.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn relocate_memory_empty_relocation_rules() {
        let mut memory = memory![((0, 0), 1), ((0, 1), 2), ((0, 2), 3)];

        assert_eq!(memory.relocate_memory(), Ok(()));
        check_memory!(memory, ((0, 0), 1), ((0, 1), 2), ((0, 2), 3));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn relocate_memory_new_segment_with_gap() {
        let mut memory = memory![
            ((0, 0), 1),
            ((0, 1), (-1, 0)),
            ((0, 2), 3),
            ((1, 0), (-1, 1)),
            ((1, 1), 5),
            ((1, 2), (-1, 2)),
            ((-1, 0), 7),
            ((-1, 1), 8),
            ((-1, 2), 9)
        ];
        memory
            .add_relocation_rule((-1, 0).into(), (2, 1).into())
            .unwrap();
        memory.data.push(vec![]);

        assert_eq!(memory.relocate_memory(), Ok(()));
        check_memory!(
            memory,
            ((0, 0), 1),
            ((0, 1), (2, 1)),
            ((0, 2), 3),
            ((1, 0), (2, 2)),
            ((1, 1), 5),
            ((1, 2), (2, 3)),
            ((2, 1), 7),
            ((2, 2), 8),
            ((2, 3), 9)
        );
        assert!(memory.temp_data.is_empty());
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn relocate_memory_new_segment() {
        let mut memory = memory![
            ((0, 0), 1),
            ((0, 1), (-1, 0)),
            ((0, 2), 3),
            ((1, 0), (-1, 1)),
            ((1, 1), 5),
            ((1, 2), (-1, 2)),
            ((-1, 0), 7),
            ((-1, 1), 8),
            ((-1, 2), 9)
        ];
        memory
            .add_relocation_rule((-1, 0).into(), (2, 0).into())
            .unwrap();
        memory.data.push(vec![]);

        assert_eq!(memory.relocate_memory(), Ok(()));

        check_memory!(
            memory,
            ((0, 0), 1),
            ((0, 1), (2, 0)),
            ((0, 2), 3),
            ((1, 0), (2, 1)),
            ((1, 1), 5),
            ((1, 2), (2, 2)),
            ((2, 0), 7),
            ((2, 1), 8),
            ((2, 2), 9)
        );
        assert!(memory.temp_data.is_empty());
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn relocate_memory_new_segment_unallocated() {
        let mut memory = memory![
            ((0, 0), 1),
            ((0, 1), (-1, 0)),
            ((0, 2), 3),
            ((1, 0), (-1, 1)),
            ((1, 1), 5),
            ((1, 2), (-1, 2)),
            ((-1, 0), 7),
            ((-1, 1), 8),
            ((-1, 2), 9)
        ];
        memory
            .add_relocation_rule((-1, 0).into(), (2, 0).into())
            .unwrap();

        assert_eq!(
            memory.relocate_memory(),
            Err(MemoryError::UnallocatedSegment(Box::new((2, 2))))
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn relocate_memory_into_existing_segment() {
        let mut memory = memory![
            ((0, 0), 1),
            ((0, 1), (-1, 0)),
            ((0, 2), 3),
            ((1, 0), (-1, 1)),
            ((1, 1), 5),
            ((1, 2), (-1, 2)),
            ((-1, 0), 7),
            ((-1, 1), 8),
            ((-1, 2), 9)
        ];
        memory
            .add_relocation_rule((-1, 0).into(), (1, 3).into())
            .unwrap();

        assert_eq!(memory.relocate_memory(), Ok(()));

        check_memory!(
            memory,
            ((0, 0), 1),
            ((0, 1), (1, 3)),
            ((0, 2), 3),
            ((1, 0), (1, 4)),
            ((1, 1), 5),
            ((1, 2), (1, 5)),
            ((1, 3), 7),
            ((1, 4), 8),
            ((1, 5), 9)
        );
        assert!(memory.temp_data.is_empty());
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn relocate_memory_into_existing_segment_inconsistent_memory() {
        let mut memory = memory![
            ((0, 0), 1),
            ((0, 1), (-1, 0)),
            ((0, 2), 3),
            ((1, 0), (-1, 1)),
            ((1, 1), 5),
            ((1, 2), (-1, 2)),
            ((-1, 0), 7),
            ((-1, 1), 8),
            ((-1, 2), 9)
        ];
        memory
            .add_relocation_rule((-1, 0).into(), (1, 0).into())
            .unwrap();

        assert_eq!(
            memory.relocate_memory(),
            Err(MemoryError::InconsistentMemory(Box::new((
                (1, 0).into(),
                (1, 1).into(),
                7.into(),
            ))))
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn relocate_memory_new_segment_2_temporary_segments_one_relocated() {
        let mut memory = memory![
            ((0, 0), 1),
            ((0, 1), (-1, 0)),
            ((0, 2), 3),
            ((1, 0), (-1, 1)),
            ((1, 1), 5),
            ((1, 2), (-1, 2)),
            ((-1, 0), 7),
            ((-1, 1), 8),
            ((-1, 2), 9),
            ((-2, 0), 10),
            ((-2, 1), 11)
        ];
        memory
            .add_relocation_rule((-1, 0).into(), (2, 0).into())
            .unwrap();
        memory.data.push(vec![]);

        assert_eq!(memory.relocate_memory(), Ok(()));
        check_memory!(
            memory,
            ((0, 0), 1),
            ((0, 1), (2, 0)),
            ((0, 2), 3),
            ((1, 0), (2, 1)),
            ((1, 1), 5),
            ((1, 2), (2, 2)),
            ((2, 0), 7),
            ((2, 1), 8),
            ((2, 2), 9),
            ((-1, 0), 10),
            ((-1, 1), 11)
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn relocate_memory_new_segment_2_temporary_segments_relocated() {
        let mut memory = memory![
            ((0, 0), 1),
            ((0, 1), (-1, 0)),
            ((0, 2), 3),
            ((1, 0), (-1, 1)),
            ((1, 1), 5),
            ((1, 2), (-1, 2)),
            ((-1, 0), 7),
            ((-1, 1), 8),
            ((-1, 2), 9),
            ((-2, 0), 10),
            ((-2, 1), 11)
        ];
        memory.data.push(vec![]);
        memory
            .add_relocation_rule((-1, 0).into(), (2, 0).into())
            .unwrap();
        memory.data.push(vec![]);
        memory
            .add_relocation_rule((-2, 0).into(), (3, 0).into())
            .unwrap();

        assert_eq!(memory.relocate_memory(), Ok(()));

        check_memory!(
            memory,
            ((0, 0), 1),
            ((0, 1), (2, 0)),
            ((0, 2), 3),
            ((1, 0), (2, 1)),
            ((1, 1), 5),
            ((1, 2), (2, 2)),
            ((2, 0), 7),
            ((2, 1), 8),
            ((2, 2), 9),
            ((3, 0), 10),
            ((3, 1), 11)
        );
        assert!(memory.temp_data.is_empty());
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_memory_display() {
        let memory = memory![
            ((0, 0), 1),
            ((0, 1), (-1, 0)),
            ((0, 2), 3),
            ((1, 0), (-1, 1)),
            ((1, 1), 5),
            ((1, 2), (-1, 2)),
            ((-1, 0), (-1, 0)),
            ((-1, 1), 8),
            ((-1, 2), 9)
        ];

        assert_eq!(
            format!("{}", memory),
            "(-1,0) : -1:0\n(-1,1) : 8\n(-1,2) : 9\n(0,0) : 1\n(0,1) : -1:0\n(0,2) : 3\n(1,0) : -1:1\n(1,1) : 5\n(1,2) : -1:2\n}\n");
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn relocate_memory_into_existing_segment_temporary_values_in_temporary_memory() {
        let mut memory = memory![
            ((0, 0), 1),
            ((0, 1), (-1, 0)),
            ((0, 2), 3),
            ((1, 0), (-1, 1)),
            ((1, 1), 5),
            ((1, 2), (-1, 2)),
            ((-1, 0), (-1, 0)),
            ((-1, 1), 8),
            ((-1, 2), 9)
        ];
        memory
            .add_relocation_rule((-1, 0).into(), (1, 3).into())
            .unwrap();

        assert_eq!(memory.relocate_memory(), Ok(()));
        check_memory!(
            memory,
            ((0, 0), 1),
            ((0, 1), (1, 3)),
            ((0, 2), 3),
            ((1, 0), (1, 4)),
            ((1, 1), 5),
            ((1, 2), (1, 5)),
            ((1, 3), (1, 3)),
            ((1, 4), 8),
            ((1, 5), 9)
        );
        assert!(memory.temp_data.is_empty());
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn relocate_address_with_rules() {
        let mut memory = Memory::new();
        memory
            .add_relocation_rule((-1, 0).into(), (2, 0).into())
            .unwrap();
        memory
            .add_relocation_rule((-2, 0).into(), (2, 2).into())
            .unwrap();

        assert_eq!(
            Memory::relocate_address((-1, 0).into(), &memory.relocation_rules),
            MaybeRelocatable::RelocatableValue((2, 0).into()),
        );
        assert_eq!(
            Memory::relocate_address((-2, 1).into(), &memory.relocation_rules),
            MaybeRelocatable::RelocatableValue((2, 3).into()),
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn relocate_address_no_rules() {
        let memory = Memory::new();
        assert_eq!(
            Memory::relocate_address((-1, 0).into(), &memory.relocation_rules),
            MaybeRelocatable::RelocatableValue((-1, 0).into()),
        );
        assert_eq!(
            Memory::relocate_address((-2, 1).into(), &memory.relocation_rules),
            MaybeRelocatable::RelocatableValue((-2, 1).into()),
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn relocate_address_real_addr() {
        let memory = Memory::new();
        assert_eq!(
            Memory::relocate_address((1, 0).into(), &memory.relocation_rules),
            MaybeRelocatable::RelocatableValue((1, 0).into()),
        );
        assert_eq!(
            Memory::relocate_address((1, 1).into(), &memory.relocation_rules),
            MaybeRelocatable::RelocatableValue((1, 1).into()),
        );
    }

    #[test]
    fn mark_address_as_accessed() {
        let mut memory = memory![((0, 0), 0)];
        assert!(!memory.data[0][0].as_ref().unwrap().is_accessed());
        memory.mark_as_accessed(relocatable!(0, 0));
        assert!(memory.data[0][0].as_ref().unwrap().is_accessed());
    }

    #[test]
    fn get_amount_of_accessed_addresses_for_segment_valid() {
        let mut memory = memory![((0, 0), 0)];
        assert_eq!(
            memory.get_amount_of_accessed_addresses_for_segment(0),
            Some(0)
        );
        memory.mark_as_accessed(relocatable!(0, 0));
        assert_eq!(
            memory.get_amount_of_accessed_addresses_for_segment(0),
            Some(1)
        );
    }

    #[test]
    fn get_amount_of_accessed_addresses_for_segment_invalid_segment() {
        let memory = memory![((0, 0), 0)];
        assert_eq!(memory.get_amount_of_accessed_addresses_for_segment(1), None);
    }

    #[test]
    fn memory_cell_new_is_not_accessed() {
        let cell = MemoryCell::new(mayberelocatable!(1));
        assert!(!cell.is_accessed())
    }

    #[test]
    fn memory_cell_mark_accessed() {
        let mut cell = MemoryCell::new(mayberelocatable!(1));
        cell.mark_accessed();
        assert!(cell.is_accessed())
    }

    #[test]
    fn memory_cell_get_value() {
        let cell = MemoryCell::new(mayberelocatable!(1));
        assert_eq!(cell.get_value(), &mayberelocatable!(1));
    }

    #[test]
    fn memory_cell_mutate_value() {
        let mut cell = MemoryCell::new(mayberelocatable!(1));
        let cell_value = cell.get_value_mut();
        assert_eq!(cell_value, &mayberelocatable!(1));
        *cell_value = mayberelocatable!(2);
        assert_eq!(cell.get_value(), &mayberelocatable!(2));
    }

    use core::cmp::Ordering::*;

    fn check_memcmp(
        lhs: (isize, usize),
        rhs: (isize, usize),
        len: usize,
        ord: Ordering,
        pos: usize,
    ) {
        let mem = memory![
            ((-2, 0), 1),
            ((-2, 1), (1, 1)),
            ((-2, 3), 0),
            ((-2, 4), 0),
            ((-1, 0), 1),
            ((-1, 1), (1, 1)),
            ((-1, 3), 0),
            ((-1, 4), 3),
            ((0, 0), 1),
            ((0, 1), (1, 1)),
            ((0, 3), 0),
            ((0, 4), 0),
            ((1, 0), 1),
            ((1, 1), (1, 1)),
            ((1, 3), 0),
            ((1, 4), 3)
        ];
        assert_eq!((ord, pos), mem.memcmp(lhs.into(), rhs.into(), len));
    }

    #[test]
    fn insert_alloc_fails_gracefully() {
        let mut mem = memory![((0, 0), 1)];
        let err = mem.insert((0, usize::MAX >> 1).into(), Felt252::one());
        assert_eq!(err, Err(MemoryError::VecCapacityExceeded));
    }

    #[test]
    fn insert_overflow_fails_gracefully() {
        let mut mem = memory![((0, 0), 1)];
        let err = mem.insert((0, usize::MAX).into(), Felt252::one());
        assert_eq!(err, Err(MemoryError::VecCapacityExceeded));
    }

    #[test]
    fn memcmp() {
        check_memcmp((0, 0), (0, 0), 3, Equal, 3);
        check_memcmp((0, 0), (1, 0), 3, Equal, 3);
        check_memcmp((0, 0), (1, 0), 5, Less, 4);
        check_memcmp((1, 0), (0, 0), 5, Greater, 4);
        check_memcmp((2, 2), (2, 5), 8, Equal, 0);
        check_memcmp((0, 0), (2, 5), 8, Greater, 0);
        check_memcmp((2, 5), (0, 0), 8, Less, 0);
        check_memcmp((-2, 0), (-2, 0), 3, Equal, 3);
        check_memcmp((-2, 0), (-1, 0), 3, Equal, 3);
        check_memcmp((-2, 0), (-1, 0), 5, Less, 4);
        check_memcmp((-1, 0), (-2, 0), 5, Greater, 4);
        check_memcmp((-3, 2), (-3, 5), 8, Equal, 0);
        check_memcmp((-2, 0), (-3, 5), 8, Greater, 0);
        check_memcmp((-3, 5), (-2, 0), 8, Less, 0);
    }
}