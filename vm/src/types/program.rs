use crate::serde::deserialize_program::{parse_program, ProgramJson, Reference, ValueAddress};
use crate::stdlib::{collections::HashMap, prelude::*, sync::Arc};

#[cfg(feature = "cairo-1-hints")]
use crate::serde::deserialize_program::{ApTracking, FlowTrackingData, ReferenceIds};
use crate::{
    hint_processor::hint_processor_definition::HintReference,
    serde::deserialize_program::{
        deserialize_and_parse_program, Attribute, BuiltinName, HintParams, Identifier,
        InstructionLocation, ReferenceManager,
    },
    types::{errors::program_errors::ProgramError, relocatable::MaybeRelocatable},
};
#[cfg(feature = "cairo-1-hints")]
use cairo_lang_casm_contract_class::CasmContractClass;
use felt::{Felt252, PRIME_STR};

#[cfg(feature = "scale-codec")]
use parity_scale_codec::{Decode, Encode, Error, Input, Output};

#[cfg(feature = "std")]
use std::path::Path;

// NOTE: `Program` has been split in two containing some data that will be deep-copied
// and some that will be allocated on the heap inside an `Arc<_>`.
// This is because it has been reported that cloning the whole structure when creating
// a `CairoRunner` becomes a bottleneck, but the following solutions were tried and
// discarded:
// - Store only a reference in `CairoRunner` rather than cloning; this doesn't work
//   because then we need to introduce explicit lifetimes, which broke `cairo-rs-py`
//   since PyO3 doesn't support Python objects containing structures with lifetimes.
// - Directly pass an `Arc<Program>` to `CairoRunner::new()` and simply copy that:
//   there was a prohibitive performance hit of 10-15% when doing so, most likely
//   either because of branch mispredictions or the extra level of indirection going
//   through a random location on the heap rather than the likely-to-be-cached spot
//   on the stack.
//
// So, the compromise was to identify which data was less used and avoid copying that,
// using `Arc<_>`, while the most accessed fields remain on the stack for the main
// loop to access. The fields in `SharedProgramData` are either preprocessed and
// copied explicitly (_in addition_ to the clone of `Program`) or are used only in
// exceptional circumstances, such as when reconstructing a backtrace on execution
// failures.
// Fields in `Program` (other than `SharedProgramData` itself) are used by the main logic.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "scale-codec", derive(Decode, Encode))]
pub(crate) struct SharedProgramData {
    pub(crate) data: Vec<MaybeRelocatable>,
    pub(crate) hints: Hints,
    pub(crate) main: Option<u64>,
    //start and end labels will only be used in proof-mode
    pub(crate) start: Option<u64>,
    pub(crate) end: Option<u64>,
    pub(crate) error_message_attributes: Vec<Attribute>,
    pub(crate) instruction_locations: Option<InstructionLocations>,
    pub(crate) identifiers: Identifiers,
    pub(crate) reference_manager: Vec<HintReference>,
}

#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct Identifiers(HashMap<String, Identifier>);
impl Identifiers {
    pub fn new() -> Self {
        Self(HashMap::new())
    }
    pub fn inner(&self) -> HashMap<String, Identifier> {
        self.0.clone()
    }
    pub fn inner_ref(&self) -> &HashMap<String, Identifier> {
        &self.0
    }
}

impl From<HashMap<String, Identifier>> for Identifiers {
    fn from(value: HashMap<String, Identifier>) -> Self {
        Self(value)
    }
}

/// SCALE trait.
#[cfg(feature = "scale-codec")]
impl Encode for Identifiers {
    fn encode_to<T: Output + ?Sized>(&self, dest: &mut T) {
        // Convert the Identifiers to Vec<(String, Identifier)> to be
        // able to use the Encode trait from this type.
        let val: Vec<(String, Identifier)> = self.0.clone().into_iter().collect();
        dest.write(&Encode::encode(&val));
    }
}
/// SCALE trait.
#[cfg(feature = "scale-codec")]
impl Decode for Identifiers {
    fn decode<I: Input>(input: &mut I) -> Result<Self, Error> {
        // Convert the Identifiers to Vec<(String, Identifier)> to be
        // able to use the Decode trait from this type.
        let val: Vec<(String, Identifier)> = Decode::decode(input)
            .map_err(|_| Error::from("Can't get EntrypointMap from input buffer."))?;
        Ok(Identifiers(HashMap::from_iter(val.into_iter())))
    }
}

#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct Hints(HashMap<u64, Vec<HintParams>>);
impl Hints {
    pub fn new() -> Self {
        Self(HashMap::new())
    }
    pub fn inner(&self) -> HashMap<u64, Vec<HintParams>> {
        self.0.clone()
    }
}

impl From<HashMap<u64, Vec<HintParams>>> for Hints {
    fn from(value: HashMap<u64, Vec<HintParams>>) -> Self {
        Self(value)
    }
}

/// SCALE trait.
#[cfg(feature = "scale-codec")]
impl Encode for Hints {
    fn encode_to<T: Output + ?Sized>(&self, dest: &mut T) {
        // Convert the Hints to Vec<(u64, Vec<HintParams>)> to be
        // able to use the Encode trait from this type.
        let val: Vec<(u64, Vec<HintParams>)> = self.0.clone().into_iter().collect();
        dest.write(&Encode::encode(&val));
    }
}
/// SCALE trait.
#[cfg(feature = "scale-codec")]
impl Decode for Hints {
    fn decode<I: Input>(input: &mut I) -> Result<Self, Error> {
        // Convert the Hints to Vec<(u64, Vec<HintParams>)> to be
        // able to use the Decode trait from this type.
        let val: Vec<(u64, Vec<HintParams>)> = Decode::decode(input)
            .map_err(|_| Error::from("Can't get EntrypointMap from input buffer."))?;
        Ok(Hints(HashMap::from_iter(val.into_iter())))
    }
}

#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct InstructionLocations(HashMap<u64, InstructionLocation>);
impl InstructionLocations {
    pub fn new() -> Self {
        Self(HashMap::new())
    }
    pub fn inner(&self) -> HashMap<u64, InstructionLocation> {
        self.0.clone()
    }
}

impl From<HashMap<u64, InstructionLocation>> for InstructionLocations {
    fn from(value: HashMap<u64, InstructionLocation>) -> Self {
        Self(value)
    }
}

/// SCALE trait.
#[cfg(feature = "scale-codec")]
impl Encode for InstructionLocations {
    fn encode_to<T: Output + ?Sized>(&self, dest: &mut T) {
        // Convert the InstructionLocations to Vec<(u64, InstructionLocation)> to be
        // able to use the Encode trait from this type.
        let val: Vec<(u64, InstructionLocation)> = self.0.clone().into_iter().collect();
        dest.write(&Encode::encode(&val));
    }
}
/// SCALE trait.
#[cfg(feature = "scale-codec")]
impl Decode for InstructionLocations {
    fn decode<I: Input>(input: &mut I) -> Result<Self, Error> {
        // Convert the InstructionLocations to Vec<(u64, InstructionLocation)> to be
        // able to use the Decode trait from this type.
        let val: Vec<(u64, InstructionLocation)> = Decode::decode(input)
            .map_err(|_| Error::from("Can't get EntrypointMap from input buffer."))?;
        Ok(InstructionLocations(HashMap::from_iter(val.into_iter())))
    }
}

#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct Constants(HashMap<String, Felt252>);
impl Constants {
    pub fn new() -> Self {
        Self(HashMap::new())
    }
    pub fn inner(&self) -> HashMap<String, Felt252> {
        self.0.clone()
    }
    pub fn inner_ref(&self) -> &HashMap<String, Felt252> {
        &self.0
    }
}

impl From<HashMap<String, Felt252>> for Constants {
    fn from(value: HashMap<String, Felt252>) -> Self {
        Self(value)
    }
}

/// SCALE trait.
#[cfg(feature = "scale-codec")]
impl Encode for Constants {
    fn encode_to<T: Output + ?Sized>(&self, dest: &mut T) {
        // Convert the Constants to Vec<(String, Felt252)> to be
        // able to use the Encode trait from this type.
        let val: Vec<(String, Felt252)> = self.0.clone().into_iter().collect();
        dest.write(&Encode::encode(&val));
    }
}
/// SCALE trait.
#[cfg(feature = "scale-codec")]
impl Decode for Constants {
    fn decode<I: Input>(input: &mut I) -> Result<Self, Error> {
        // Convert the Constants to Vec<(String, Felt252)> to be
        // able to use the Decode trait from this type.
        let val: Vec<(String, Felt252)> = Decode::decode(input)
            .map_err(|_| Error::from("Can't get EntrypointMap from input buffer."))?;
        Ok(Constants(HashMap::from_iter(val.into_iter())))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "scale-codec", derive(Decode, Encode))]
pub struct Program {
    pub(crate) shared_program_data: Arc<SharedProgramData>,
    pub(crate) constants: Constants,
    pub(crate) builtins: Vec<BuiltinName>,
}

impl Program {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        builtins: Vec<BuiltinName>,
        data: Vec<MaybeRelocatable>,
        main: Option<u64>,
        hints: HashMap<u64, Vec<HintParams>>,
        reference_manager: ReferenceManager,
        identifiers: HashMap<String, Identifier>,
        error_message_attributes: Vec<Attribute>,
        instruction_locations: Option<HashMap<u64, InstructionLocation>>,
    ) -> Result<Program, ProgramError> {
        let mut constants = HashMap::new();
        for (key, value) in identifiers.iter() {
            if value.type_.as_deref() == Some("const") {
                let value = value
                    .value
                    .clone()
                    .ok_or_else(|| ProgramError::ConstWithoutValue(key.clone()))?;
                constants.insert(key.clone(), value);
            }
        }
        let shared_program_data = SharedProgramData {
            data,
            hints: Hints::from(hints),
            main,
            start: None,
            end: None,
            error_message_attributes,
            instruction_locations: instruction_locations.map(InstructionLocations::from),
            identifiers: Identifiers::from(identifiers),
            reference_manager: Self::get_reference_list(&reference_manager),
        };
        Ok(Self {
            shared_program_data: Arc::new(shared_program_data),
            constants: Constants::from(constants),
            builtins,
        })
    }

    #[cfg(feature = "std")]
    pub fn from_file(path: &Path, entrypoint: Option<&str>) -> Result<Program, ProgramError> {
        let file_content = std::fs::read(path)?;
        deserialize_and_parse_program(&file_content, entrypoint)
    }

    pub fn from_bytes(bytes: &[u8], entrypoint: Option<&str>) -> Result<Program, ProgramError> {
        deserialize_and_parse_program(bytes, entrypoint)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let program_json: ProgramJson = parse_program(self.clone());
        serde_json::to_vec(&program_json).unwrap()
    }

    pub fn prime(&self) -> &str {
        _ = self;
        PRIME_STR
    }

    pub fn iter_builtins(&self) -> impl Iterator<Item = &BuiltinName> {
        self.builtins.iter()
    }

    pub fn iter_data(&self) -> impl Iterator<Item = &MaybeRelocatable> {
        self.shared_program_data.data.iter()
    }

    pub fn data_len(&self) -> usize {
        self.shared_program_data.data.len()
    }

    pub fn builtins_len(&self) -> usize {
        self.builtins.len()
    }

    pub fn get_identifier(&self, id: &str) -> Option<&Identifier> {
        self.shared_program_data.identifiers.inner_ref().get(id)
    }

    pub fn iter_identifiers(&self) -> impl Iterator<Item = (&str, &Identifier)> {
        self.shared_program_data
            .identifiers
            .inner_ref()
            .iter()
            .map(|(cairo_type, identifier)| (cairo_type.as_str(), identifier))
    }

    pub(crate) fn get_reference_list(reference_manager: &ReferenceManager) -> Vec<HintReference> {
        reference_manager
            .references
            .iter()
            .map(|r| r.to_owned().into())
            .collect()
    }
}

impl Program {
    pub fn builtins(&self) -> &Vec<BuiltinName> {
        &self.builtins
    }

    pub fn data(&self) -> &Vec<MaybeRelocatable> {
        &self.shared_program_data.data
    }

    pub fn hints(&self) -> &Hints {
        &self.shared_program_data.hints
    }

    pub fn main(&self) -> &Option<u64> {
        &self.shared_program_data.main
    }

    pub fn start(&self) -> &Option<u64> {
        &self.shared_program_data.start
    }

    pub fn end(&self) -> &Option<u64> {
        &self.shared_program_data.end
    }

    pub fn error_message_attributes(&self) -> &Vec<Attribute> {
        &self.shared_program_data.error_message_attributes
    }

    pub fn instruction_locations(&self) -> &Option<InstructionLocations> {
        &self.shared_program_data.instruction_locations
    }

    pub fn identifiers(&self) -> &Identifiers {
        &self.shared_program_data.identifiers
    }

    pub fn constants(&self) -> &Constants {
        &self.constants
    }

    pub fn reference_manager(&self) -> ReferenceManager {
        ReferenceManager {
            references: self
                .shared_program_data
                .reference_manager
                .iter()
                .map(|r| Reference {
                    value_address: ValueAddress {
                        offset1: r.offset1.clone(),
                        offset2: r.offset2.clone(),
                        dereference: r.dereference,
                        value_type: r.cairo_type.clone().unwrap_or_default(),
                    },
                    ap_tracking_data: r.ap_tracking_data.clone().unwrap_or_default(),
                    pc: r.pc,
                })
                .collect(),
        }
    }
}

impl Default for Program {
    fn default() -> Self {
        Self {
            shared_program_data: Arc::new(SharedProgramData::default()),
            constants: Constants::new(),
            builtins: Vec::new(),
        }
    }
}

#[cfg(feature = "cairo-1-hints")]
// Note: This Program will only work when using run_from_entrypoint, and the Cairo1Hintprocesso
impl TryFrom<CasmContractClass> for Program {
    type Error = ProgramError;
    fn try_from(value: CasmContractClass) -> Result<Self, ProgramError> {
        let data = value
            .bytecode
            .iter()
            .map(|x| MaybeRelocatable::from(Felt252::from(&x.value)))
            .collect();
        // Hint data is going to be hosted processor-side, hints field will only store the pc where hints are located.
        // Only one pc will be stored, so the hint processor will be responsible for executing all hints for a given pc
        let hints = value
            .hints
            .iter()
            .map(|(x, _)| {
                (
                    *x as u64,
                    vec![HintParams {
                        code: x.to_string(),
                        accessible_scopes: Vec::new(),
                        flow_tracking_data: FlowTrackingData {
                            ap_tracking: ApTracking::default(),
                            reference_ids: ReferenceIds::new(),
                        },
                    }],
                )
            })
            .collect();
        let error_message_attributes = Vec::new();
        let reference_manager = ReferenceManager {
            references: Vec::new(),
        };
        Self::new(
            vec![],
            data,
            None,
            hints,
            reference_manager,
            HashMap::new(),
            error_message_attributes,
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serde::deserialize_program::{ApTracking, FlowTrackingData, Members};
    use crate::utils::test_utils::*;
    use felt::felt_str;
    use num_traits::Zero;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    #[test]
    fn test_serialize_program() {
        let reference_manager = ReferenceManager {
            references: Vec::new(),
        };

        let builtins: Vec<BuiltinName> = Vec::new();
        let data: Vec<MaybeRelocatable> = vec![
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(1000),
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(2000),
            mayberelocatable!(5201798304953696256),
            mayberelocatable!(2345108766317314046),
        ];

        let program = Program::new(
            builtins,
            data,
            None,
            HashMap::new(),
            reference_manager,
            HashMap::new(),
            Vec::new(),
            None,
        )
        .unwrap();

        assert_eq!(
            program,
            Program::from_bytes(&program.to_bytes(), None).unwrap()
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn new() {
        let reference_manager = ReferenceManager {
            references: Vec::new(),
        };

        let builtins: Vec<BuiltinName> = Vec::new();
        let data: Vec<MaybeRelocatable> = vec![
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(1000),
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(2000),
            mayberelocatable!(5201798304953696256),
            mayberelocatable!(2345108766317314046),
        ];

        let program = Program::new(
            builtins.clone(),
            data.clone(),
            None,
            HashMap::new(),
            reference_manager,
            HashMap::new(),
            Vec::new(),
            None,
        )
        .unwrap();

        assert_eq!(program.builtins, builtins);
        assert_eq!(program.shared_program_data.data, data);
        assert_eq!(program.shared_program_data.main, None);
        assert_eq!(program.shared_program_data.identifiers, Identifiers::new());
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn new_program_with_identifiers() {
        let reference_manager = ReferenceManager {
            references: Vec::new(),
        };

        let builtins: Vec<BuiltinName> = Vec::new();

        let data: Vec<MaybeRelocatable> = vec![
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(1000),
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(2000),
            mayberelocatable!(5201798304953696256),
            mayberelocatable!(2345108766317314046),
        ];

        let mut identifiers: HashMap<String, Identifier> = HashMap::new();

        identifiers.insert(
            String::from("__main__.main"),
            Identifier {
                pc: Some(0),
                type_: Some(String::from("function")),
                value: None,
                full_name: None,
                members: None,
                cairo_type: None,
                decorators: None,
                size: None,
                destination: None,
                references: None,
            },
        );

        identifiers.insert(
            String::from("__main__.main.SIZEOF_LOCALS"),
            Identifier {
                pc: None,
                type_: Some(String::from("const")),
                value: Some(Felt252::zero()),
                full_name: None,
                members: None,
                cairo_type: None,
                decorators: None,
                size: None,
                destination: None,
                references: None,
            },
        );

        let program = Program::new(
            builtins.clone(),
            data.clone(),
            None,
            HashMap::new(),
            reference_manager,
            identifiers.clone(),
            Vec::new(),
            None,
        )
        .unwrap();

        assert_eq!(program.builtins, builtins);
        assert_eq!(program.shared_program_data.data, data);
        assert_eq!(program.shared_program_data.main, None);
        assert_eq!(
            program.shared_program_data.identifiers,
            Identifiers::from(identifiers)
        );
        assert_eq!(
            program.constants,
            Constants::from(
                [("__main__.main.SIZEOF_LOCALS", Felt252::zero())]
                    .into_iter()
                    .map(|(key, value)| (key.to_string(), value))
                    .collect::<HashMap<_, _>>()
            ),
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn get_prime() {
        let program = Program::default();
        assert_eq!(PRIME_STR, program.prime());
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn iter_builtins() {
        let reference_manager = ReferenceManager {
            references: Vec::new(),
        };

        let builtins: Vec<_> = vec![BuiltinName::range_check, BuiltinName::bitwise];
        let data: Vec<_> = vec![
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(1000),
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(2000),
            mayberelocatable!(5201798304953696256),
            mayberelocatable!(2345108766317314046),
        ];

        let program = Program::new(
            builtins.clone(),
            data,
            None,
            HashMap::new(),
            reference_manager,
            HashMap::new(),
            Vec::new(),
            None,
        )
        .unwrap();

        assert_eq!(
            program.iter_builtins().cloned().collect::<Vec<_>>(),
            builtins
        );

        assert_eq!(program.builtins_len(), 2);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn iter_data() {
        let reference_manager = ReferenceManager {
            references: Vec::new(),
        };

        let builtins: Vec<BuiltinName> = Vec::new();
        let data: Vec<MaybeRelocatable> = vec![
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(1000),
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(2000),
            mayberelocatable!(5201798304953696256),
            mayberelocatable!(2345108766317314046),
        ];

        let program = Program::new(
            builtins,
            data.clone(),
            None,
            HashMap::new(),
            reference_manager,
            HashMap::new(),
            Vec::new(),
            None,
        )
        .unwrap();

        assert_eq!(program.iter_data().cloned().collect::<Vec<_>>(), data);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn data_len() {
        let reference_manager = ReferenceManager {
            references: Vec::new(),
        };

        let builtins: Vec<BuiltinName> = Vec::new();
        let data: Vec<MaybeRelocatable> = vec![
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(1000),
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(2000),
            mayberelocatable!(5201798304953696256),
            mayberelocatable!(2345108766317314046),
        ];

        let program = Program::new(
            builtins,
            data.clone(),
            None,
            HashMap::new(),
            reference_manager,
            HashMap::new(),
            Vec::new(),
            None,
        )
        .unwrap();

        assert_eq!(program.data_len(), data.len());
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn get_identifier() {
        let reference_manager = ReferenceManager {
            references: Vec::new(),
        };

        let builtins: Vec<BuiltinName> = Vec::new();

        let data: Vec<MaybeRelocatable> = vec![
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(1000),
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(2000),
            mayberelocatable!(5201798304953696256),
            mayberelocatable!(2345108766317314046),
        ];

        let mut identifiers: HashMap<String, Identifier> = HashMap::new();

        identifiers.insert(
            String::from("__main__.main"),
            Identifier {
                pc: Some(0),
                type_: Some(String::from("function")),
                value: None,
                full_name: None,
                members: None,
                cairo_type: None,
                decorators: None,
                size: None,
                destination: None,
                references: None,
            },
        );

        identifiers.insert(
            String::from("__main__.main.SIZEOF_LOCALS"),
            Identifier {
                pc: None,
                type_: Some(String::from("const")),
                value: Some(Felt252::zero()),
                full_name: None,
                members: None,
                cairo_type: None,
                decorators: None,
                size: None,
                destination: None,
                references: None,
            },
        );

        let program = Program::new(
            builtins,
            data,
            None,
            HashMap::new(),
            reference_manager,
            identifiers.clone(),
            Vec::new(),
            None,
        )
        .unwrap();

        assert_eq!(
            program.get_identifier("__main__.main"),
            identifiers.get("__main__.main"),
        );
        assert_eq!(
            program.get_identifier("__main__.main.SIZEOF_LOCALS"),
            identifiers.get("__main__.main.SIZEOF_LOCALS"),
        );
        assert_eq!(
            program.get_identifier("missing"),
            identifiers.get("missing"),
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn iter_identifiers() {
        let reference_manager = ReferenceManager {
            references: Vec::new(),
        };

        let builtins: Vec<BuiltinName> = Vec::new();

        let data: Vec<MaybeRelocatable> = vec![
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(1000),
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(2000),
            mayberelocatable!(5201798304953696256),
            mayberelocatable!(2345108766317314046),
        ];

        let mut identifiers: HashMap<String, Identifier> = HashMap::new();

        identifiers.insert(
            String::from("__main__.main"),
            Identifier {
                pc: Some(0),
                type_: Some(String::from("function")),
                decorators: None,
                destination: None,
                references: None,
                size: None,
                value: None,
                full_name: None,
                members: None,
                cairo_type: None,
            },
        );

        identifiers.insert(
            String::from("__main__.main.SIZEOF_LOCALS"),
            Identifier {
                pc: None,
                type_: Some(String::from("const")),
                value: Some(Felt252::zero()),
                decorators: None,
                destination: None,
                references: None,
                size: None,
                full_name: None,
                members: None,
                cairo_type: None,
            },
        );

        let program = Program::new(
            builtins,
            data,
            None,
            HashMap::new(),
            reference_manager,
            identifiers.clone(),
            Vec::new(),
            None,
        )
        .unwrap();

        let collected_identifiers: HashMap<_, _> = program
            .iter_identifiers()
            .map(|(cairo_type, identifier)| (cairo_type.to_string(), identifier.clone()))
            .collect();

        assert_eq!(collected_identifiers, identifiers);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn new_program_with_invalid_identifiers() {
        let reference_manager = ReferenceManager {
            references: Vec::new(),
        };

        let builtins: Vec<BuiltinName> = Vec::new();

        let data: Vec<MaybeRelocatable> = vec![
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(1000),
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(2000),
            mayberelocatable!(5201798304953696256),
            mayberelocatable!(2345108766317314046),
        ];

        let mut identifiers: HashMap<String, Identifier> = HashMap::new();

        identifiers.insert(
            String::from("__main__.main"),
            Identifier {
                pc: Some(0),
                type_: Some(String::from("function")),
                value: None,
                full_name: None,
                members: None,
                cairo_type: None,
                decorators: None,
                size: None,
                destination: None,
                references: None,
            },
        );

        identifiers.insert(
            String::from("__main__.main.SIZEOF_LOCALS"),
            Identifier {
                pc: None,
                type_: Some(String::from("const")),
                value: None,
                full_name: None,
                members: None,
                cairo_type: None,
                decorators: None,
                size: None,
                destination: None,
                references: None,
            },
        );

        let program = Program::new(
            builtins,
            data,
            None,
            HashMap::new(),
            reference_manager,
            identifiers.clone(),
            Vec::new(),
            None,
        );

        assert!(program.is_err());
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn deserialize_program_test() {
        let program = Program::from_bytes(
            include_bytes!("../../../cairo_programs/manually_compiled/valid_program_a.json"),
            Some("main"),
        )
        .unwrap();

        let builtins: Vec<BuiltinName> = Vec::new();
        let data: Vec<MaybeRelocatable> = vec![
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(1000),
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(2000),
            mayberelocatable!(5201798304953696256),
            mayberelocatable!(2345108766317314046),
        ];

        let mut identifiers: HashMap<String, Identifier> = HashMap::new();

        identifiers.insert(
            String::from("__main__.main"),
            Identifier {
                pc: Some(0),
                type_: Some(String::from("function")),
                value: None,
                full_name: None,
                members: None,
                cairo_type: None,
                decorators: Some(vec![]),
                size: None,
                destination: None,
                references: None,
            },
        );
        identifiers.insert(
            String::from("__main__.main.Args"),
            Identifier {
                pc: None,
                type_: Some(String::from("struct")),
                value: None,
                full_name: Some("__main__.main.Args".to_string()),
                members: Some(Members::new()),
                cairo_type: None,
                decorators: None,
                size: Some(0),
                destination: None,
                references: None,
            },
        );
        identifiers.insert(
            String::from("__main__.main.ImplicitArgs"),
            Identifier {
                pc: None,
                type_: Some(String::from("struct")),
                value: None,
                full_name: Some("__main__.main.ImplicitArgs".to_string()),
                members: Some(Members::new()),
                cairo_type: None,
                decorators: None,
                size: Some(0),
                destination: None,
                references: None,
            },
        );
        identifiers.insert(
            String::from("__main__.main.Return"),
            Identifier {
                pc: None,
                type_: Some(String::from("struct")),
                value: None,
                full_name: Some("__main__.main.Return".to_string()),
                members: Some(Members::new()),
                cairo_type: None,
                decorators: None,
                size: Some(0),
                destination: None,
                references: None,
            },
        );
        identifiers.insert(
            String::from("__main__.main.SIZEOF_LOCALS"),
            Identifier {
                pc: None,
                type_: Some(String::from("const")),
                value: Some(Felt252::zero()),
                full_name: None,
                members: None,
                cairo_type: None,
                decorators: None,
                size: None,
                destination: None,
                references: None,
            },
        );

        assert_eq!(program.builtins, builtins);
        assert_eq!(program.shared_program_data.data, data);
        assert_eq!(program.shared_program_data.main, Some(0));
        assert_eq!(
            program.shared_program_data.identifiers,
            Identifiers::from(identifiers)
        );
    }

    /// Deserialize a program without an entrypoint.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn deserialize_program_without_entrypoint_test() {
        let program = Program::from_bytes(
            include_bytes!("../../../cairo_programs/manually_compiled/valid_program_a.json"),
            None,
        )
        .unwrap();

        let builtins: Vec<BuiltinName> = Vec::new();

        let error_message_attributes: Vec<Attribute> = vec![Attribute {
            name: String::from("error_message"),
            start_pc: 379,
            end_pc: 381,
            value: String::from("SafeUint256: addition overflow"),
            flow_tracking_data: Some(FlowTrackingData {
                ap_tracking: ApTracking {
                    group: 14,
                    offset: 35,
                },
                reference_ids: ReferenceIds::new(),
            }),
            accessible_scopes: vec![
                "openzeppelin.security.safemath.library".to_string(),
                "openzeppelin.security.safemath.library.SafeUint256".to_string(),
                "openzeppelin.security.safemath.library.SafeUint256.add".to_string(),
            ],
        }];

        let data: Vec<MaybeRelocatable> = vec![
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(1000),
            mayberelocatable!(5189976364521848832),
            mayberelocatable!(2000),
            mayberelocatable!(5201798304953696256),
            mayberelocatable!(2345108766317314046),
        ];

        let mut identifiers: HashMap<String, Identifier> = HashMap::new();

        identifiers.insert(
            String::from("__main__.main"),
            Identifier {
                pc: Some(0),
                type_: Some(String::from("function")),
                value: None,
                full_name: None,
                members: None,
                cairo_type: None,
                decorators: Some(vec![]),
                size: None,
                destination: None,
                references: None,
            },
        );
        identifiers.insert(
            String::from("__main__.main.Args"),
            Identifier {
                pc: None,
                type_: Some(String::from("struct")),
                value: None,
                full_name: Some("__main__.main.Args".to_string()),
                members: Some(Members::new()),
                cairo_type: None,
                decorators: None,
                size: Some(0),
                destination: None,
                references: None,
            },
        );
        identifiers.insert(
            String::from("__main__.main.ImplicitArgs"),
            Identifier {
                pc: None,
                type_: Some(String::from("struct")),
                value: None,
                full_name: Some("__main__.main.ImplicitArgs".to_string()),
                members: Some(Members::new()),
                cairo_type: None,
                decorators: None,
                size: Some(0),
                destination: None,
                references: None,
            },
        );
        identifiers.insert(
            String::from("__main__.main.Return"),
            Identifier {
                pc: None,
                type_: Some(String::from("struct")),
                value: None,
                full_name: Some("__main__.main.Return".to_string()),
                members: Some(Members::new()),
                cairo_type: None,
                decorators: None,
                size: Some(0),
                destination: None,
                references: None,
            },
        );
        identifiers.insert(
            String::from("__main__.main.SIZEOF_LOCALS"),
            Identifier {
                pc: None,
                type_: Some(String::from("const")),
                value: Some(Felt252::zero()),
                full_name: None,
                members: None,
                cairo_type: None,
                decorators: None,
                size: None,
                destination: None,
                references: None,
            },
        );

        assert_eq!(program.builtins, builtins);
        assert_eq!(program.shared_program_data.data, data);
        assert_eq!(program.shared_program_data.main, None);
        assert_eq!(
            program.shared_program_data.identifiers,
            Identifiers::from(identifiers)
        );
        assert_eq!(
            program.shared_program_data.error_message_attributes,
            error_message_attributes
        )
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn deserialize_program_constants_test() {
        let program = Program::from_bytes(
            include_bytes!(
                "../../../cairo_programs/manually_compiled/deserialize_constant_test.json"
            ),
            Some("main"),
        )
        .unwrap();

        let constants = [
            ("__main__.compare_abs_arrays.SIZEOF_LOCALS", Felt252::zero()),
            (
                "starkware.cairo.common.cairo_keccak.packed_keccak.ALL_ONES",
                felt_str!(
                    "3618502788666131106986593281521497120414687020801267626233049500247285301247"
                ),
            ),
            (
                "starkware.cairo.common.cairo_keccak.packed_keccak.BLOCK_SIZE",
                Felt252::new(3),
            ),
            (
                "starkware.cairo.common.alloc.alloc.SIZEOF_LOCALS",
                felt_str!(
                    "-3618502788666131213697322783095070105623107215331596699973092056135872020481"
                ),
            ),
            (
                "starkware.cairo.common.uint256.SHIFT",
                felt_str!("340282366920938463463374607431768211456"),
            ),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect::<HashMap<_, _>>();

        assert_eq!(program.constants, Constants::from(constants));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn default_program() {
        let shared_program_data = SharedProgramData {
            data: Vec::new(),
            hints: Hints::new(),
            main: None,
            start: None,
            end: None,
            error_message_attributes: Vec::new(),
            instruction_locations: None,
            identifiers: Identifiers::new(),
            reference_manager: Program::get_reference_list(&ReferenceManager {
                references: Vec::new(),
            }),
        };
        let program = Program {
            shared_program_data: Arc::new(shared_program_data),
            constants: Constants::new(),
            builtins: Vec::new(),
        };

        assert_eq!(program, Program::default());
    }
}
