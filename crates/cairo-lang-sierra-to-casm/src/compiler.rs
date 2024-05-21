use cairo_lang_casm::inline::CasmContext;
use cairo_lang_sierra::extensions::bitwise::BitwiseType;
use cairo_lang_sierra::extensions::ec::EcOpType;
use cairo_lang_sierra::extensions::pedersen::PedersenType;
use cairo_lang_sierra::extensions::poseidon::PoseidonType;
use cairo_lang_sierra::extensions::range_check::{RangeCheck96Type, RangeCheckType};
use cairo_lang_sierra::extensions::segment_arena::SegmentArenaType;
use cairo_lang_sierra::extensions::starknet::syscalls::SystemType;
use serde::ser::Error as SerError;
use serde::ser::SerializeSeq;
use serde::{Serialize, Serializer};
use std::collections::{HashMap, HashSet};
use std::fmt::Display;

use cairo_felt::Felt252;
use cairo_lang_casm::assembler::AssembledCairoProgram;
use cairo_lang_casm::instructions::{Instruction, InstructionBody, RetInstruction};
use cairo_lang_casm::{casm, casm_extend};
use cairo_lang_sierra::extensions::circuit::{
    AddModType, CircuitConcreteLibfunc, CircuitInfo, MulModType, VALUE_SIZE,
};
use cairo_lang_sierra::extensions::const_type::ConstConcreteLibfunc;
use cairo_lang_sierra::extensions::core::{
    CoreConcreteLibfunc, CoreLibfunc, CoreType, CoreTypeConcrete,
};
use cairo_lang_sierra::extensions::coupon::CouponConcreteLibfunc;
use cairo_lang_sierra::extensions::gas::{GasBuiltinType, GasConcreteLibfunc};
use cairo_lang_sierra::extensions::lib_func::SierraApChange;
use cairo_lang_sierra::extensions::ConcreteLibfunc;
use cairo_lang_sierra::extensions::{ConcreteType, NamedType};
use cairo_lang_sierra::ids::{ConcreteLibfuncId, ConcreteTypeId, GenericTypeId, VarId};
use cairo_lang_sierra::program::{
    BranchTarget, GenericArg, Invocation, Program, Statement, StatementIdx,
};
use cairo_lang_sierra::program_registry::{ProgramRegistry, ProgramRegistryError};
use cairo_lang_sierra_type_size::{get_type_size_map, TypeSizeMap};
use cairo_lang_utils::casts::IntoOrPanic;
use cairo_lang_utils::ordered_hash_map::OrderedHashMap;
use cairo_lang_utils::unordered_hash_map::UnorderedHashMap;
use cairo_lang_utils::unordered_hash_set::UnorderedHashSet;
use itertools::{chain, zip_eq};
use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};
use thiserror::Error;

use crate::annotations::{AnnotationError, ProgramAnnotations, StatementAnnotations};
use crate::circuit::CircuitsInfo;
use crate::invocations::enm::get_variant_selector;
use crate::invocations::{
    check_references_on_stack, compile_invocation, BranchChanges, InvocationError, ProgramInfo,
};
use crate::metadata::Metadata;
use crate::references::{check_types_match, ReferenceValue, ReferencesError};
use crate::relocations::{relocate_instructions, RelocationEntry};

#[cfg(test)]
#[path = "compiler_test.rs"]
mod test;

#[derive(Error, Debug, Eq, PartialEq)]
pub enum CompilationError {
    #[error("Failed building type information")]
    FailedBuildingTypeInformation,
    #[error("Error from program registry: {0}")]
    ProgramRegistryError(Box<ProgramRegistryError>),
    #[error(transparent)]
    AnnotationError(#[from] AnnotationError),
    #[error("#{statement_idx}: {error}")]
    InvocationError { statement_idx: StatementIdx, error: InvocationError },
    #[error("#{statement_idx}: Return arguments are not on the stack.")]
    ReturnArgumentsNotOnStack { statement_idx: StatementIdx },
    #[error("#{statement_idx}: {error}")]
    ReferencesError { statement_idx: StatementIdx, error: ReferencesError },
    #[error("#{statement_idx}: Invocation mismatched to libfunc")]
    LibfuncInvocationMismatch { statement_idx: StatementIdx },
    #[error("{var_id} is dangling at #{statement_idx}.")]
    DanglingReferences { statement_idx: StatementIdx, var_id: VarId },
    #[error("#{source_statement_idx}->#{destination_statement_idx}: Expected branch align")]
    ExpectedBranchAlign {
        source_statement_idx: StatementIdx,
        destination_statement_idx: StatementIdx,
    },
    #[error("Const data does not match the declared const type.")]
    ConstDataMismatch,
    #[error("Unsupported const type.")]
    UnsupportedConstType,
    #[error("Unsupported circuit type.")]
    UnsupportedCircuitType,
    #[error("Const segments must appear in ascending order without holes.")]
    ConstSegmentsOutOfOrder,
    #[error("Code size limit exceeded.")]
    CodeSizeLimitExceeded,
    #[error("Unknown function id in metadata.")]
    MetadataUnknownFunctionId,
    #[error("Statement #{0} out of bounds in metadata.")]
    MetadataStatementOutOfBound(StatementIdx),
    #[error("Statement #{0} should not have gas variables.")]
    StatementNotSupportingGasVariables(StatementIdx),
    #[error("Statement #{0} should not have ap-change variables.")]
    StatementNotSupportingApChangeVariables(StatementIdx),
    #[error("Expected all gas variables to be positive.")]
    MetadataNegativeGasVariable,
    #[error("Function param {param_index} only partially contains argument {arg_index}.")]
    ArgumentUnaligned { param_index: usize, arg_index: usize },
    #[error("Function expects arguments of size {expected} and received {actual} instead.")]
    ArgumentsSizeMismatch { expected: usize, actual: usize },
}

/// Configuration for the Sierra to CASM compilation.
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub struct SierraToCasmConfig {
    /// Whether to check the gas usage of the program.
    pub gas_usage_check: bool,
    /// CASM bytecode size limit.
    pub max_bytecode_size: usize,
}

/// The casm program representation.
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct CairoProgram {
    pub instructions: Vec<Instruction>,
    pub debug_info: CairoProgramDebugInfo,
    pub consts_info: ConstsInfo,
}
impl Display for CairoProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if std::env::var("PRINT_CASM_BYTECODE_OFFSETS").is_ok() {
            let mut bytecode_offset = 0;
            for instruction in &self.instructions {
                writeln!(f, "{instruction}; // {bytecode_offset}")?;
                bytecode_offset += instruction.body.op_size();
            }
            for segment in self.consts_info.segments.values() {
                writeln!(f, "ret; // {bytecode_offset}")?;
                bytecode_offset += 1;
                for value in &segment.values {
                    writeln!(f, "dw {value}; // {bytecode_offset}")?;
                    bytecode_offset += 1;
                }
            }
        } else {
            for instruction in &self.instructions {
                writeln!(f, "{instruction};")?;
            }
            for segment in self.consts_info.segments.values() {
                writeln!(f, "ret;")?;
                for value in &segment.values {
                    writeln!(f, "dw {value};")?;
                }
            }
        }
        Ok(())
    }
}

impl CairoProgram {
    /// Creates an assembled representation of the program.
    pub fn assemble(&self) -> AssembledCairoProgram {
        self.assemble_ex(&[], &[])
    }

    /// Creates an assembled representation of the program preceded by `header` and followed by
    /// `footer`.
    pub fn assemble_ex(
        &self,
        header: &[Instruction],
        footer: &[Instruction],
    ) -> AssembledCairoProgram {
        let mut bytecode = vec![];
        let mut hints = vec![];
        for instruction in chain!(header, &self.instructions) {
            if !instruction.hints.is_empty() {
                hints.push((bytecode.len(), instruction.hints.clone()))
            }
            bytecode.extend(instruction.assemble().encode().into_iter())
        }
        let [ref ret_bytecode] = Instruction::new(InstructionBody::Ret(RetInstruction {}), false)
            .assemble()
            .encode()[..]
        else {
            panic!("`ret` instruction should be a single word.")
        };
        for segment in self.consts_info.segments.values() {
            bytecode.push(ret_bytecode.clone());
            bytecode.extend(segment.values.clone());
        }
        for instruction in footer {
            assert!(
                instruction.hints.is_empty(),
                "All footer instructions must have no hints since these cannot be added to the \
                 hints dict."
            );
            bytecode.extend(instruction.assemble().encode().into_iter())
        }
        AssembledCairoProgram { bytecode, hints }
    }
}

pub struct CairoProgramWithContext<'a> {
    pub cairo_program: &'a CairoProgram,
    pub sierra_program: &'a Program,
    pub metadata: &'a Metadata,
}

impl<'a> CairoProgramWithContext<'a> {
    pub fn new(
        cairo_program: &'a CairoProgram,
        sierra_program: &'a Program,
        metadata: &'a Metadata,
    ) -> CairoProgramWithContext<'a> {
        CairoProgramWithContext { cairo_program, sierra_program, metadata }
    }
}

impl<'a> Serialize for CairoProgramWithContext<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        let initial_gas = 0_usize;
        let main_func = self
            .sierra_program
            .find_function("::main")
            .ok_or_else(|| S::Error::custom("Main function not found"))?;
        let params = &main_func
            .signature
            .param_types
            .iter()
            .map(|pt| {
                let sierra_program_registry =
                    ProgramRegistry::<CoreType, CoreLibfunc>::new_with_ap_change(
                        &self.sierra_program,
                        self.metadata.ap_change_info.function_ap_change.clone(),
                    )
                    .map_err(|_| S::Error::custom("Failed building type information"))
                    .unwrap();
                let info = sierra_program_registry.get_type(pt).unwrap().info();
                let generic_id = &info.long_id.generic_id;
                let type_sizes = get_type_size_map(&self.sierra_program, &sierra_program_registry)
                    .ok_or_else(|| S::Error::custom("Failed building type information"))
                    .unwrap();
                let size = type_sizes[pt];
                (generic_id.clone(), size)
            })
            .collect::<Vec<(GenericTypeId, i16)>>();

        let entry_point = main_func.entry_point.0;
        let code_offset =
            self.cairo_program.debug_info.sierra_statement_info[entry_point].start_offset;

        let (entry_code, builtins) =
            create_entry_code_from_params(&params, &[], initial_gas, code_offset)
                .map_err(|_| S::Error::custom("Failed building entry code"))
                .unwrap();
        let entry_code_assembled: Vec<BigInt> =
            entry_code.iter().flat_map(|instruction| instruction.assemble().encode()).collect();
        let builtin_names = builtins.iter().map(|builtin| builtin.name()).collect::<Vec<&str>>();
        let footer = create_code_footer();
        // TODO: build hints map
        let hints = chain!(&entry_code, &self.cairo_program.instructions);
        let assembled_cairo_program = self.cairo_program.assemble_ex(&entry_code, &footer);
        map.serialize_entry("bytecode", &assembled_cairo_program.bytecode)?;
        map.serialize_entry("hints", &assembled_cairo_program.hints)?;
        // TODO: Add debug info.
        // map.serialize_entry("debug_info", &self.cairo_program.debug_info)?;
        // TODO: Add entry point
        map.serialize_entry("entry_code", &entry_code_assembled)?;
        map.serialize_entry("builtins", &builtin_names)?;
        // TODO: Add builtins
        // map.serialize_entry("builtins", builtins)?;
        // TODO: add hints
        // map.serialize_entry("hints", hints)?;
        map.serialize_entry("consts_info", &self.cairo_program.consts_info)?;
        map.end()
    }
}

pub fn create_entry_code_from_params(
    param_types: &[(GenericTypeId, i16)],
    args: &[Arg],
    initial_gas: usize,
    code_offset: usize,
) -> Result<(Vec<Instruction>, Vec<BuiltinName>), CompilationError> {
    let mut ctx = casm! {};
    // The builtins in the formatting expected by the runner.
    let builtins = vec![
        BuiltinName::pedersen,
        BuiltinName::range_check,
        BuiltinName::bitwise,
        BuiltinName::ec_op,
        BuiltinName::poseidon,
    ];
    // The offset [fp - i] for each of this builtins in this configuration.
    let builtin_offset: HashMap<GenericTypeId, i16> = HashMap::from([
        (PedersenType::ID, 7),
        (RangeCheckType::ID, 6),
        (BitwiseType::ID, 5),
        (EcOpType::ID, 4),
        (PoseidonType::ID, 3),
    ]);

    let emulated_builtins = HashSet::from([
        SystemType::ID,
        // TODO(ilya): Move to following when supported by cairo-vm.
        RangeCheck96Type::ID,
        AddModType::ID,
        MulModType::ID,
    ]);

    let mut ap_offset: i16 = 0;
    let mut array_args_data_iter = prep_array_args(&mut ctx, args, &mut ap_offset).into_iter();
    let after_arrays_data_offset = ap_offset;
    if param_types.iter().any(|(ty, _)| ty == &SegmentArenaType::ID) {
        casm_extend! {ctx,
            // SegmentArena segment.
            %{ memory[ap + 0] = segments.add() %}
            // Infos segment.
            %{ memory[ap + 1] = segments.add() %}
            ap += 2;
            [ap + 0] = 0, ap++;
            // Write Infos segment, n_constructed (0), and n_destructed (0) to the segment.
            [ap - 2] = [[ap - 3]];
            [ap - 1] = [[ap - 3] + 1];
            [ap - 1] = [[ap - 3] + 2];
        }
        ap_offset += 3;
    }
    let mut expected_arguments_size = 0;
    let mut param_index = 0;
    let mut arg_iter = args.iter().enumerate();
    for ty in param_types {
        let (generic_ty, ty_size) = ty;
        if let Some(offset) = builtin_offset.get(generic_ty) {
            casm_extend! {ctx,
                [ap + 0] = [fp - offset], ap++;
            }
            ap_offset += 1;
        } else if emulated_builtins.contains(generic_ty) {
            casm_extend! {ctx,
                %{ memory[ap + 0] = segments.add() %}
                ap += 1;
            }
            ap_offset += 1;
        } else if generic_ty == &GasBuiltinType::ID {
            casm_extend! {ctx,
                [ap + 0] = initial_gas, ap++;
            }
            ap_offset += 1;
        } else if generic_ty == &SegmentArenaType::ID {
            let offset = -ap_offset + after_arrays_data_offset;
            casm_extend! {ctx,
                [ap + 0] = [ap + offset] + 3, ap++;
            }
            ap_offset += 1;
        } else {
            let arg_size = *ty_size;
            let param_ap_offset_end = ap_offset + arg_size;
            expected_arguments_size += arg_size.into_or_panic::<usize>();
            while ap_offset < param_ap_offset_end {
                let Some((arg_index, arg)) = arg_iter.next() else {
                    break;
                };
                add_arg_to_stack(&mut ctx, arg, &mut ap_offset, &mut array_args_data_iter);
                if ap_offset > param_ap_offset_end {
                    return Err(CompilationError::ArgumentUnaligned { param_index, arg_index });
                }
            }
            param_index += 1;
        };
    }
    let actual_args_size = args
        .iter()
        .map(|arg| match arg {
            Arg::Value(_) => 1,
            Arg::Array(_) => 2,
        })
        .sum::<usize>();
    if expected_arguments_size != actual_args_size {
        return Err(CompilationError::ArgumentsSizeMismatch {
            expected: expected_arguments_size,
            actual: actual_args_size,
        });
    }
    let before_final_call = ctx.current_code_offset;
    let final_call_size = 3;
    let offset = final_call_size + code_offset;
    casm_extend! {ctx,
        call rel offset;
        ret;
    }
    assert_eq!(before_final_call + final_call_size, ctx.current_code_offset);
    Ok((ctx.instructions, builtins))
}

fn prep_array_args(ctx: &mut CasmContext, args: &[Arg], ap_offset: &mut i16) -> Vec<ArrayDataInfo> {
    let mut array_args_data = vec![];
    for arg in args {
        let Arg::Array(values) = arg else { continue };
        let mut inner_array_args_data = prep_array_args(ctx, values, ap_offset).into_iter();
        casm_extend! {ctx,
            %{ memory[ap + 0] = segments.add() %}
            ap += 1;
        }

        let ptr_offset = *ap_offset;
        *ap_offset += 1;
        let data_offset = *ap_offset;
        for arg in values {
            add_arg_to_stack(ctx, arg, ap_offset, &mut inner_array_args_data);
        }
        let ptr = *ap_offset - ptr_offset;
        let size = *ap_offset - data_offset;
        for i in 0..size {
            casm_extend! {ctx, [ap + (i - size)] = [[ap - ptr] + i]; }
        }
        array_args_data.push(ArrayDataInfo { ptr_offset, size });
    }
    array_args_data
}

/// The information on an array argument that was added to the stack.
struct ArrayDataInfo {
    /// The offset of the pointer to the array data in the stack.
    ptr_offset: i16,
    /// The size of the array data in the stack.
    size: i16,
}

// TODO: remove repitiion. This is already defined in the cairo_vm
enum BuiltinName {
    output,
    range_check,
    pedersen,
    ecdsa,
    keccak,
    bitwise,
    ec_op,
    poseidon,
    segment_arena,
}

impl BuiltinName {
    pub fn name(&self) -> &'static str {
        match self {
            BuiltinName::output => OUTPUT_BUILTIN_NAME,
            BuiltinName::range_check => RANGE_CHECK_BUILTIN_NAME,
            BuiltinName::pedersen => HASH_BUILTIN_NAME,
            BuiltinName::ecdsa => SIGNATURE_BUILTIN_NAME,
            BuiltinName::keccak => KECCAK_BUILTIN_NAME,
            BuiltinName::bitwise => BITWISE_BUILTIN_NAME,
            BuiltinName::ec_op => EC_OP_BUILTIN_NAME,
            BuiltinName::poseidon => POSEIDON_BUILTIN_NAME,
            BuiltinName::segment_arena => SEGMENT_ARENA_BUILTIN_NAME,
        }
    }
}

// TODO: remove repitition. This is already defined in the cairo_vm
pub const OUTPUT_BUILTIN_NAME: &str = "output_builtin";
pub const HASH_BUILTIN_NAME: &str = "pedersen_builtin";
pub const RANGE_CHECK_BUILTIN_NAME: &str = "range_check_builtin";
pub const SIGNATURE_BUILTIN_NAME: &str = "ecdsa_builtin";
pub const BITWISE_BUILTIN_NAME: &str = "bitwise_builtin";
pub const EC_OP_BUILTIN_NAME: &str = "ec_op_builtin";
pub const KECCAK_BUILTIN_NAME: &str = "keccak_builtin";
pub const POSEIDON_BUILTIN_NAME: &str = "poseidon_builtin";
pub const SEGMENT_ARENA_BUILTIN_NAME: &str = "segment_arena_builtin";

// TODO: remove repitition. This is already defined in the cairo-runner
/// An argument to a sierra function run,
#[derive(Debug)]
pub enum Arg {
    Value(Felt252),
    Array(Vec<Arg>),
}
impl From<Felt252> for Arg {
    fn from(value: Felt252) -> Self {
        Self::Value(value)
    }
}

/// Adds an argument to the stack, updating the ap_offset and the array_data_iter.
fn add_arg_to_stack(
    ctx: &mut CasmContext,
    arg: &Arg,
    ap_offset: &mut i16,
    array_data_iter: &mut impl Iterator<Item = ArrayDataInfo>,
) {
    match arg {
        Arg::Value(value) => {
            casm_extend! {ctx,
                [ap + 0] = (value.to_bigint()), ap++;
            }
            *ap_offset += 1;
        }
        Arg::Array(_) => {
            let info = array_data_iter.next().unwrap();
            casm_extend! {ctx,
                [ap + 0] = [ap + (info.ptr_offset - *ap_offset)], ap++;
                [ap + 0] = [ap - 1] + (info.size), ap++;
            }
            *ap_offset += 2;
        }
    }
}

/// The debug information of a compilation from Sierra to casm.
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct SierraStatementDebugInfo {
    /// The start offset of the sierra statement within the bytecode.
    pub start_offset: usize,
    /// The end offset of the sierra statement within the bytecode.
    pub end_offset: usize,
    /// The index of the sierra statement in the instructions vector.
    pub instruction_idx: usize,
    /// Statement-kind-dependent information.
    pub additional_kind_info: StatementKindDebugInfo,
}

/// Additional debug information for a Sierra statement, depending on its kind
/// (invoke/return/dummy).
#[derive(Debug, Eq, PartialEq, Clone)]
pub enum StatementKindDebugInfo {
    Return(ReturnStatementDebugInfo),
    Invoke(InvokeStatementDebugInfo),
}

/// Additional debug information for a return Sierra statement.
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct ReturnStatementDebugInfo {
    /// The references of a Sierra return statement.
    pub ref_values: Vec<ReferenceValue>,
}

/// Additional debug information for an invoke Sierra statement.
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct InvokeStatementDebugInfo {
    /// The result branch changes of a Sierra invoke statement.
    pub result_branch_changes: Vec<BranchChanges>,
    /// The references of a Sierra invoke statement.
    pub ref_values: Vec<ReferenceValue>,
}

/// The debug information of a compilation from Sierra to casm.
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct CairoProgramDebugInfo {
    /// The debug information per Sierra statement.
    pub sierra_statement_info: Vec<SierraStatementDebugInfo>,
}

/// The information about the constants used in the program.
#[derive(Serialize, Debug, Eq, PartialEq, Default, Clone)]
pub struct ConstsInfo {
    pub segments: OrderedHashMap<u32, ConstSegment>,
    pub total_segments_size: usize,

    /// Maps a circuit to its segment id.
    pub circuit_segments: OrderedHashMap<ConcreteTypeId, u32>,
}
impl ConstsInfo {
    /// Creates a new `ConstSegmentsInfo` from the given libfuncs.
    pub fn new<'a>(
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        type_sizes: &TypeSizeMap,
        libfunc_ids: impl Iterator<Item = &'a ConcreteLibfuncId> + Clone,
        circuit_infos: &OrderedHashMap<ConcreteTypeId, CircuitInfo>,
        const_segments_max_size: usize,
    ) -> Result<Self, CompilationError> {
        let mut segments_data_size = 0;

        // A lambda to add a const.
        // Note that `segments` is passed as an argument to avoid taking borrowing it.
        let mut add_const = |segments: &mut OrderedHashMap<u32, ConstSegment>,
                             segment_id,
                             ty,
                             const_data: Vec<BigInt>| {
            let segment: &mut ConstSegment = segments.entry(segment_id).or_default();

            segments_data_size += const_data.len();
            segment.const_offset.insert(ty, segment.values.len());
            segment.values.extend(const_data);
            if segments_data_size + segments.len() > const_segments_max_size {
                return Err(CompilationError::CodeSizeLimitExceeded);
            }
            Ok(())
        };

        let mut segments = OrderedHashMap::default();

        for id in libfunc_ids.clone() {
            if let CoreConcreteLibfunc::Const(ConstConcreteLibfunc::AsBox(as_box)) =
                registry.get_libfunc(id).unwrap()
            {
                add_const(
                    &mut segments,
                    as_box.segment_id,
                    as_box.const_type.clone(),
                    extract_const_value(registry, type_sizes, &as_box.const_type).unwrap(),
                )?;
            }
        }

        // Check that the segments were declared in order and without holes.
        if segments
            .keys()
            .enumerate()
            .any(|(i, segment_id)| i != segment_id.into_or_panic::<usize>())
        {
            return Err(CompilationError::ConstSegmentsOutOfOrder);
        }

        let mut next_segment = segments.len() as u32;
        let mut circuit_segments = OrderedHashMap::default();

        for id in libfunc_ids {
            if let CoreConcreteLibfunc::Circuit(CircuitConcreteLibfunc::GetDescriptor(libfunc)) =
                registry.get_libfunc(id).unwrap()
            {
                let circ_ty = &libfunc.ty;
                let info = circuit_infos.get(circ_ty).unwrap();
                let mut const_value: Vec<BigInt> = vec![];
                let mut push_offset =
                    |offset: usize| const_value.push((offset * VALUE_SIZE).into());
                for gate_offsets in chain!(info.add_offsets.iter(), info.mul_offsets.iter()) {
                    push_offset(gate_offsets.lhs);
                    push_offset(gate_offsets.rhs);
                    push_offset(gate_offsets.output);
                }

                add_const(&mut segments, next_segment, circ_ty.clone(), const_value)?;
                circuit_segments.insert(circ_ty.clone(), next_segment);
                next_segment += 1;
            }
        }

        let mut total_segments_size = 0;
        for (_, segment) in segments.iter_mut() {
            segment.segment_offset = total_segments_size;
            // Add 1 for the `ret` instruction.
            total_segments_size += 1 + segment.values.len();
        }
        Ok(Self { segments, total_segments_size, circuit_segments })
    }
}

/// The data for a single segment.
#[derive(Serialize, Debug, Eq, PartialEq, Default, Clone)]
pub struct ConstSegment {
    /// The values in the segment.
    #[serde(serialize_with = "serialize_bigint_vec")]
    pub values: Vec<BigInt>,
    /// The offset of each const within the segment.
    pub const_offset: UnorderedHashMap<ConcreteTypeId, usize>,
    /// The offset of the segment relative to the end of the code segment.
    pub segment_offset: usize,
}

fn serialize_bigint_vec<S>(bigint_vec: &Vec<BigInt>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut seq = serializer.serialize_seq(Some(bigint_vec.len()))?;
    for bigint in bigint_vec {
        let string_repr = bigint.to_string();
        seq.serialize_element(&string_repr)?;
    }
    seq.end()
}

/// Gets a concrete type, if it is a const type returns a vector of the values to be stored in the
/// const segment.
fn extract_const_value(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    type_sizes: &TypeSizeMap,
    ty: &ConcreteTypeId,
) -> Result<Vec<BigInt>, CompilationError> {
    let mut values = Vec::new();
    let mut types_stack = vec![ty.clone()];
    while let Some(ty) = types_stack.pop() {
        let CoreTypeConcrete::Const(const_type) = registry.get_type(&ty).unwrap() else {
            return Err(CompilationError::UnsupportedConstType);
        };
        let inner_type = registry.get_type(&const_type.inner_ty).unwrap();
        match inner_type {
            CoreTypeConcrete::Struct(_) => {
                // Add the struct members' types to the stack in reverse order.
                for arg in const_type.inner_data.iter().rev() {
                    match arg {
                        GenericArg::Type(arg_ty) => types_stack.push(arg_ty.clone()),
                        _ => return Err(CompilationError::ConstDataMismatch),
                    }
                }
            }
            CoreTypeConcrete::Enum(enm) => {
                // The first argument is the variant selector, the second is the variant data.
                match &const_type.inner_data[..] {
                    [GenericArg::Value(variant_index), GenericArg::Type(ty)] => {
                        let variant_index = variant_index.to_usize().unwrap();
                        values.push(
                            get_variant_selector(enm.variants.len(), variant_index).unwrap().into(),
                        );
                        let full_enum_size: usize =
                            type_sizes[&const_type.inner_ty].into_or_panic();
                        let variant_size: usize =
                            type_sizes[&enm.variants[variant_index]].into_or_panic();
                        // Padding with zeros to full enum size.
                        values.extend(itertools::repeat_n(
                            BigInt::zero(),
                            // Subtract 1 due to the variant selector.
                            full_enum_size - variant_size - 1,
                        ));
                        types_stack.push(ty.clone());
                    }
                    _ => return Err(CompilationError::ConstDataMismatch),
                }
            }
            CoreTypeConcrete::NonZero(_) => match &const_type.inner_data[..] {
                [GenericArg::Type(inner)] => {
                    types_stack.push(inner.clone());
                }
                _ => return Err(CompilationError::ConstDataMismatch),
            },
            _ => match &const_type.inner_data[..] {
                [GenericArg::Value(value)] => {
                    values.push(value.clone());
                }
                _ => return Err(CompilationError::ConstDataMismatch),
            },
        };
    }
    Ok(values)
}

/// Creates a list of instructions that will be appended to the program's bytecode.
pub fn create_code_footer() -> Vec<Instruction> {
    casm! {
        // Add a `ret` instruction used in libfuncs that retrieve the current value of the `fp`
        // and `pc` registers.
        ret;
    }
    .instructions
}

/// Ensure the basic structure of the invocation is the same as the library function.
pub fn check_basic_structure(
    statement_idx: StatementIdx,
    invocation: &Invocation,
    libfunc: &CoreConcreteLibfunc,
) -> Result<(), CompilationError> {
    if invocation.args.len() != libfunc.param_signatures().len()
        || !itertools::equal(
            invocation.branches.iter().map(|branch| branch.results.len()),
            libfunc.output_types().iter().map(|types| types.len()),
        )
        || match libfunc.fallthrough() {
            Some(expected_fallthrough) => {
                invocation.branches[expected_fallthrough].target != BranchTarget::Fallthrough
            }
            None => false,
        }
    {
        Err(CompilationError::LibfuncInvocationMismatch { statement_idx })
    } else {
        Ok(())
    }
}

/// Compiles `program` from Sierra to CASM using `metadata` for information regarding AP changes
/// and gas usage, and config additional compilation flavours.
pub fn compile(
    program: &Program,
    metadata: &Metadata,
    config: SierraToCasmConfig,
) -> Result<CairoProgram, Box<CompilationError>> {
    let mut instructions = Vec::new();
    let mut relocations: Vec<RelocationEntry> = Vec::new();

    // Maps statement_idx to its debug info.
    // The last value (for statement_idx=number-of-statements)
    // contains the final offset (the size of the program code segment).
    let mut sierra_statement_info: Vec<SierraStatementDebugInfo> =
        Vec::with_capacity(program.statements.len());

    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new_with_ap_change(
        program,
        metadata.ap_change_info.function_ap_change.clone(),
    )
    .map_err(CompilationError::ProgramRegistryError)?;
    validate_metadata(program, &registry, metadata)?;
    let type_sizes = get_type_size_map(program, &registry)
        .ok_or(CompilationError::FailedBuildingTypeInformation)?;
    let mut backwards_jump_indices = UnorderedHashSet::<_>::default();
    for (statement_id, statement) in program.statements.iter().enumerate() {
        if let Statement::Invocation(invocation) = statement {
            for branch in &invocation.branches {
                if let BranchTarget::Statement(target) = branch.target {
                    if target.0 < statement_id {
                        backwards_jump_indices.insert(target);
                    }
                }
            }
        }
    }
    let mut program_annotations = ProgramAnnotations::create(
        program.statements.len(),
        backwards_jump_indices,
        &program.funcs,
        metadata,
        config.gas_usage_check,
        &type_sizes,
    )
    .map_err(|err| Box::new(err.into()))?;

    let circuits_info =
        CircuitsInfo::new(&registry, program.type_declarations.iter().map(|td| &td.id))?;

    let mut program_offset: usize = 0;
    for (statement_id, statement) in program.statements.iter().enumerate() {
        let statement_idx = StatementIdx(statement_id);

        if program_offset > config.max_bytecode_size {
            return Err(Box::new(CompilationError::CodeSizeLimitExceeded));
        }
        match statement {
            Statement::Return(ref_ids) => {
                let (annotations, return_refs) = program_annotations
                    .get_annotations_after_take_args(statement_idx, ref_ids.iter())
                    .map_err(|err| Box::new(err.into()))?;
                return_refs.iter().for_each(|r| r.validate(&type_sizes));

                if let Some(var_id) = annotations.refs.keys().next() {
                    return Err(Box::new(CompilationError::DanglingReferences {
                        statement_idx,
                        var_id: var_id.clone(),
                    }));
                };

                program_annotations
                    .validate_final_annotations(
                        statement_idx,
                        &annotations,
                        &program.funcs,
                        metadata,
                        &return_refs,
                    )
                    .map_err(|err| Box::new(err.into()))?;
                check_references_on_stack(&return_refs).map_err(|error| match error {
                    InvocationError::InvalidReferenceExpressionForArgument => {
                        CompilationError::ReturnArgumentsNotOnStack { statement_idx }
                    }
                    _ => CompilationError::InvocationError { statement_idx, error },
                })?;

                let start_offset = program_offset;

                let ret_instruction = RetInstruction {};
                program_offset += ret_instruction.op_size();

                sierra_statement_info.push(SierraStatementDebugInfo {
                    start_offset,
                    end_offset: program_offset,
                    instruction_idx: instructions.len(),
                    additional_kind_info: StatementKindDebugInfo::Return(
                        ReturnStatementDebugInfo { ref_values: return_refs },
                    ),
                });

                instructions.push(Instruction::new(InstructionBody::Ret(ret_instruction), false));
            }
            Statement::Invocation(invocation) => {
                let (annotations, invoke_refs) = program_annotations
                    .get_annotations_after_take_args(statement_idx, invocation.args.iter())
                    .map_err(|err| Box::new(err.into()))?;

                let libfunc = registry
                    .get_libfunc(&invocation.libfunc_id)
                    .map_err(CompilationError::ProgramRegistryError)?;
                check_basic_structure(statement_idx, invocation, libfunc)?;

                let param_types: Vec<_> = libfunc
                    .param_signatures()
                    .iter()
                    .map(|param_signature| param_signature.ty.clone())
                    .collect();
                check_types_match(&invoke_refs, &param_types).map_err(|error| {
                    Box::new(AnnotationError::ReferencesError { statement_idx, error }.into())
                })?;
                invoke_refs.iter().for_each(|r| r.validate(&type_sizes));
                let compiled_invocation = compile_invocation(
                    ProgramInfo {
                        metadata,
                        type_sizes: &type_sizes,
                        circuits_info: &circuits_info,
                        const_data_values: &|ty| {
                            extract_const_value(&registry, &type_sizes, ty).unwrap()
                        },
                    },
                    invocation,
                    libfunc,
                    statement_idx,
                    &invoke_refs,
                    annotations.environment,
                )
                .map_err(|error| CompilationError::InvocationError { statement_idx, error })?;

                let start_offset = program_offset;

                for instruction in &compiled_invocation.instructions {
                    program_offset += instruction.body.op_size();
                }

                sierra_statement_info.push(SierraStatementDebugInfo {
                    start_offset,
                    end_offset: program_offset,
                    instruction_idx: instructions.len(),
                    additional_kind_info: StatementKindDebugInfo::Invoke(
                        InvokeStatementDebugInfo {
                            result_branch_changes: compiled_invocation.results.clone(),
                            ref_values: invoke_refs,
                        },
                    ),
                });

                for entry in compiled_invocation.relocations {
                    relocations.push(RelocationEntry {
                        instruction_idx: instructions.len() + entry.instruction_idx,
                        relocation: entry.relocation,
                    });
                }
                instructions.extend(compiled_invocation.instructions);

                let branching_libfunc = compiled_invocation.results.len() > 1;
                // Using a vector of annotations for the loop allows us to clone the annotations
                // only in the case of more the 1 branch, which is less common.
                let mut all_updated_annotations = vec![StatementAnnotations {
                    environment: compiled_invocation.environment,
                    ..annotations
                }];
                while all_updated_annotations.len() < compiled_invocation.results.len() {
                    all_updated_annotations.push(all_updated_annotations[0].clone());
                }

                for ((branch_info, branch_changes), updated_annotations) in
                    zip_eq(&invocation.branches, compiled_invocation.results)
                        .zip(all_updated_annotations)
                {
                    let destination_statement_idx = statement_idx.next(&branch_info.target);
                    if branching_libfunc
                        && !is_branch_align(
                            &registry,
                            &program.statements[destination_statement_idx.0],
                        )?
                    {
                        return Err(Box::new(CompilationError::ExpectedBranchAlign {
                            source_statement_idx: statement_idx,
                            destination_statement_idx,
                        }));
                    }

                    program_annotations
                        .propagate_annotations(
                            statement_idx,
                            destination_statement_idx,
                            updated_annotations,
                            branch_info,
                            branch_changes,
                            branching_libfunc,
                        )
                        .map_err(|err| Box::new(err.into()))?;
                }
            }
        }
    }

    let statement_offsets: Vec<usize> = std::iter::once(0)
        .chain(sierra_statement_info.iter().map(|s: &SierraStatementDebugInfo| s.end_offset))
        .collect();

    let const_segments_max_size = config
        .max_bytecode_size
        .checked_sub(program_offset)
        .ok_or_else(|| Box::new(CompilationError::CodeSizeLimitExceeded))?;
    let consts_info = ConstsInfo::new(
        &registry,
        &type_sizes,
        program.libfunc_declarations.iter().map(|ld| &ld.id),
        &circuits_info.circuits,
        const_segments_max_size,
    )?;
    relocate_instructions(&relocations, &statement_offsets, &consts_info, &mut instructions);

    Ok(CairoProgram {
        instructions,
        consts_info,
        debug_info: CairoProgramDebugInfo { sierra_statement_info },
    })
}

/// Runs basic validations on the given metadata.
pub fn validate_metadata(
    program: &Program,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &Metadata,
) -> Result<(), CompilationError> {
    // Function validations.
    for function_id in metadata.ap_change_info.function_ap_change.keys() {
        registry
            .get_function(function_id)
            .map_err(|_| CompilationError::MetadataUnknownFunctionId)?;
    }
    for (function_id, costs) in metadata.gas_info.function_costs.iter() {
        registry
            .get_function(function_id)
            .map_err(|_| CompilationError::MetadataUnknownFunctionId)?;
        for (_token_type, value) in costs.iter() {
            if *value < 0 {
                return Err(CompilationError::MetadataNegativeGasVariable);
            }
        }
    }

    // Get the libfunc for the given statement index, or an error.
    let get_libfunc = |idx: &StatementIdx| -> Result<&CoreConcreteLibfunc, CompilationError> {
        if let Statement::Invocation(invocation) =
            program.get_statement(idx).ok_or(CompilationError::MetadataStatementOutOfBound(*idx))?
        {
            registry
                .get_libfunc(&invocation.libfunc_id)
                .map_err(CompilationError::ProgramRegistryError)
        } else {
            Err(CompilationError::StatementNotSupportingApChangeVariables(*idx))
        }
    };

    // Statement validations.
    for idx in metadata.ap_change_info.variable_values.keys() {
        if !matches!(get_libfunc(idx)?, CoreConcreteLibfunc::BranchAlign(_)) {
            return Err(CompilationError::StatementNotSupportingApChangeVariables(*idx));
        }
    }
    for ((idx, _token), value) in metadata.gas_info.variable_values.iter() {
        if *value < 0 {
            return Err(CompilationError::MetadataNegativeGasVariable);
        }
        if !matches!(
            get_libfunc(idx)?,
            CoreConcreteLibfunc::BranchAlign(_)
                | CoreConcreteLibfunc::Coupon(CouponConcreteLibfunc::Refund(_))
                | CoreConcreteLibfunc::Gas(
                    GasConcreteLibfunc::WithdrawGas(_)
                        | GasConcreteLibfunc::BuiltinWithdrawGas(_)
                        | GasConcreteLibfunc::RedepositGas(_)
                )
        ) {
            return Err(CompilationError::StatementNotSupportingGasVariables(*idx));
        }
    }
    Ok(())
}

/// Returns true if `statement` is an invocation of the branch_align libfunc.
fn is_branch_align(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    statement: &Statement,
) -> Result<bool, CompilationError> {
    if let Statement::Invocation(invocation) = statement {
        let libfunc = registry
            .get_libfunc(&invocation.libfunc_id)
            .map_err(CompilationError::ProgramRegistryError)?;
        if let [branch_signature] = libfunc.branch_signatures() {
            if branch_signature.ap_change == SierraApChange::BranchAlign {
                return Ok(true);
            }
        }
    }

    Ok(false)
}
