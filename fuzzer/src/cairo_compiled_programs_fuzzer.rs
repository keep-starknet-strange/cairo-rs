#![no_main]
use cairo_felt::Felt252;
use cairo_vm::cairo_run::{self, CairoRunConfig};
use cairo_vm::hint_processor::builtin_hint_processor::builtin_hint_processor_definition::BuiltinHintProcessor;
use libfuzzer_sys::{
    arbitrary::{Arbitrary, Unstructured},
    fuzz_target,
};
use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

// Global counter for fuzz iteration
static FUZZ_ITERATION_COUNT: AtomicUsize = AtomicUsize::new(0);

fuzz_target!(|data: (&[u8], Felt252, Felt252, Felt252, Felt252, u128)| {
    // Define fuzzer iteration with id purposes
    let _iteration_count = FUZZ_ITERATION_COUNT.fetch_add(1, Ordering::SeqCst);

    // Define default configuration
    let cairo_run_config = CairoRunConfig::default();
    let mut hint_executor = BuiltinHintProcessor::new_empty();

    let mut array = Vec::new();
    let mut unstructured = Unstructured::new(data.0);

    for _x in 0..(data.5 as u8) {
        array.push(Felt252::arbitrary(&mut unstructured).unwrap())
    }

    // Create and run the programs
    program_array_sum(&array, &cairo_run_config, &mut hint_executor);
    program_unsafe_keccak(&array, &cairo_run_config, &mut hint_executor);
    program_bitwise(&data.1, &data.2, &cairo_run_config, &mut hint_executor);
    program_poseidon(
        &data.1,
        &data.2,
        &data.3,
        &cairo_run_config,
        &mut hint_executor,
    );
    program_range_check(
        &data.1,
        &data.2,
        &data.3,
        &cairo_run_config,
        &mut hint_executor,
    );
    program_ec_op(data.5, &cairo_run_config, &mut hint_executor);
    program_pedersen(&data.1, &data.2, &cairo_run_config, &mut hint_executor);
    program_ecdsa(
        &data.1,
        &data.2,
        &data.3,
        &data.4,
        &cairo_run_config,
        &mut hint_executor,
    );
});

fn program_array_sum(
    array: &Vec<Felt252>,
    cairo_run_config: &CairoRunConfig,
    hint_executor: &mut BuiltinHintProcessor,
) {
    let populated_array = array
        .iter()
        .enumerate()
        .map(|(index, num)| format!("assert [ptr + {}] = {};  \n", index, num))
        .collect::<Vec<_>>()
        .join("            ")
        .repeat(array.len());

    let file_content = format!(
        "
        %builtins output

        from starkware.cairo.common.alloc import alloc
        from starkware.cairo.common.serialize import serialize_word
        
        // Computes the sum of the memory elements at addresses:
        //   arr + 0, arr + 1, ..., arr + (size - 1).
        func array_sum(arr: felt*, size) -> (sum: felt) {{
            if (size == 0) {{
                return (sum=0);
            }}
        
            // size is not zero.
            let (sum_of_rest) = array_sum(arr=arr + 1, size=size - 1);
            return (sum=[arr] + sum_of_rest);
        }}
        
        func main{{output_ptr: felt*}}() {{
            const ARRAY_SIZE = {};
        
            // Allocate an array.
            let (ptr) = alloc();
        
            // Populate some values in the array.
            {populated_array}
        
            // Call array_sum to compute the sum of the elements.
            let (sum) = array_sum(arr=ptr, size=ARRAY_SIZE);
        
            // Write the sum to the program output.
            serialize_word(sum);
        
            return ();
    }}
    ",
        array.len()
    );

    // Create programs names and program
    let cairo_path_array_sum = format!("cairo_programs/array_sum_{:?}.cairo", FUZZ_ITERATION_COUNT);
    let json_path_array_sum = format!("cairo_programs/array_sum_{:?}.json", FUZZ_ITERATION_COUNT);
    let _ = fs::write(&cairo_path_array_sum, file_content.as_bytes());

    compile_program(&cairo_path_array_sum, &json_path_array_sum);

    let program_content_array_sum = std::fs::read(&json_path_array_sum).unwrap();

    // Run the program with default configurations
    let _ = cairo_run::cairo_run(&program_content_array_sum, cairo_run_config, hint_executor);

    // Remove files to save memory
    delete_files(&cairo_path_array_sum, &json_path_array_sum);
}

fn program_unsafe_keccak(
    array: &Vec<Felt252>,
    cairo_run_config: &CairoRunConfig,
    hint_executor: &mut BuiltinHintProcessor,
) {
    let populated_array = array
        .iter()
        .enumerate()
        .map(|(index, num)| format!("assert data[{}] = {}; \n", index, num))
        .collect::<Vec<_>>()
        .join("            ");

    let file_content = format!(
        "
    %builtins output

    from starkware.cairo.common.alloc import alloc
    from starkware.cairo.common.serialize import serialize_word
    from starkware.cairo.common.keccak import unsafe_keccak

    func main{{output_ptr: felt*}}() {{
        alloc_locals;

        let (data: felt*) = alloc();

        {populated_array}

        let (low: felt, high: felt) = unsafe_keccak(data, {});

        serialize_word(low);
        serialize_word(high);

        return ();
    }}
    ",
        array.len()
    );

    // Create programs names and program
    let cairo_path_unsafe_keccak = format!(
        "cairo_programs/unsafe_keccak_{:?}.cairo",
        FUZZ_ITERATION_COUNT
    );
    let json_path_unsafe_keccak = format!(
        "cairo_programs/unsafe_keccak_{:?}.json",
        FUZZ_ITERATION_COUNT
    );
    let _ = fs::write(&cairo_path_unsafe_keccak, file_content.as_bytes());

    compile_program(&cairo_path_unsafe_keccak, &json_path_unsafe_keccak);

    let program_content_unsafe_keccak = std::fs::read(&json_path_unsafe_keccak).unwrap();

    // Run the program with default configurations
    let _ = cairo_run::cairo_run(
        &program_content_unsafe_keccak,
        cairo_run_config,
        hint_executor,
    );

    // Remove files to save memory
    delete_files(&cairo_path_unsafe_keccak, &json_path_unsafe_keccak);
}

fn program_bitwise(
    num1: &Felt252,
    num2: &Felt252,
    cairo_run_config: &CairoRunConfig,
    hint_executor: &mut BuiltinHintProcessor,
) {
    let and = num1 & num2;
    let xor = num1 ^ num2;
    let or = num1 | num2;
    let file_content = format!("
    %builtins bitwise
    from starkware.cairo.common.bitwise import bitwise_and, bitwise_xor, bitwise_or, bitwise_operations
    from starkware.cairo.common.cairo_builtins import BitwiseBuiltin

    func main{{bitwise_ptr: BitwiseBuiltin*}}() {{
        let (and_a) = bitwise_and({num1}, {num2});  
        assert and_a = {and}; 
        let (xor_a) = bitwise_xor(num1, num2);
        assert xor_a = {xor};
        let (or_a) = bitwise_or(num1, num2);
        assert or_a = {or};

        let (and_b, xor_b, or_b) = bitwise_operations({num1}, {num2});
        assert and_b = {and};
        assert xor_b = {xor};
        assert or_b = {or};
        return ();
    }}

    ");

    // Create programs names and program
    let cairo_path_bitwise = format!("cairo_programs/bitwise-{:?}.cairo", FUZZ_ITERATION_COUNT);
    let json_path_bitwise = format!("cairo_programs/bitwise-{:?}.json", FUZZ_ITERATION_COUNT);
    let _ = fs::write(&cairo_path_bitwise, file_content.as_bytes());

    compile_program(&cairo_path_bitwise, &json_path_bitwise);

    let program_content_bitwise = std::fs::read(&json_path_bitwise).unwrap();

    // Run the program with default configurations
    let _ = cairo_run::cairo_run(&program_content_bitwise, cairo_run_config, hint_executor);

    // Remove files to save memory
    delete_files(&cairo_path_bitwise, &json_path_bitwise);
}

fn program_poseidon(
    num1: &Felt252,
    num2: &Felt252,
    num3: &Felt252,
    cairo_run_config: &CairoRunConfig,
    hint_executor: &mut BuiltinHintProcessor,
) {
    let file_content = format!(
        "
    %builtins poseidon
    from starkware.cairo.common.cairo_builtins import PoseidonBuiltin
    from starkware.cairo.common.poseidon_state import PoseidonBuiltinState
    from starkware.cairo.common.builtin_poseidon.poseidon import (
        poseidon_hash,
        poseidon_hash_single,
        poseidon_hash_many,
    )
    from starkware.cairo.common.alloc import alloc

    func main{{poseidon_ptr: PoseidonBuiltin*}}() {{
        // Hash one
        let (x) = poseidon_hash_single(
            {num3}
        );
        // Hash two
        let (y) = poseidon_hash({num1}, {num2});
        // Hash three
        let felts: felt* = alloc();
        assert felts[0] = {num1};
        assert felts[1] = {num2};
        assert felts[2] = {num3};
        let (z) = poseidon_hash_many(3, felts);
        return ();
    }}

    "
    );

    // Create programs names and program
    let cairo_path_poseidon = format!("cairo_programs/poseidon_{:?}.cairo", FUZZ_ITERATION_COUNT);
    let json_path_poseidon = format!("cairo_programs/poseidon_{:?}.json", FUZZ_ITERATION_COUNT);
    let _ = fs::write(&cairo_path_poseidon, file_content.as_bytes());

    compile_program(&cairo_path_poseidon, &json_path_poseidon);

    let program_content_poseidon = std::fs::read(&json_path_poseidon).unwrap();

    // Run the program with default configurations
    let _ = cairo_run::cairo_run(&program_content_poseidon, cairo_run_config, hint_executor);

    // Remove files to save memory
    delete_files(&cairo_path_poseidon, &json_path_poseidon);
}

fn program_range_check(
    num1: &Felt252,
    num2: &Felt252,
    num3: &Felt252,
    cairo_run_config: &CairoRunConfig,
    hint_executor: &mut BuiltinHintProcessor,
) {
    let file_content = format!(
        "
    %builtins range_check

    from starkware.cairo.common.math import assert_250_bit
    from starkware.cairo.common.alloc import alloc
    
    func assert_250_bit_element_array{{range_check_ptr: felt}}(
        array: felt*, array_length: felt, iterator: felt
    ) {{
        if (iterator == array_length) {{
            return ();
        }}
        assert_250_bit(array[iterator]);
        return assert_250_bit_element_array(array, array_length, iterator + 1);
    }}
    
    func fill_array(array: felt*, base: felt, step: felt, array_length: felt, iterator: felt) {{
        if (iterator == array_length) {{
            return ();
        }}
        assert array[iterator] = base + step * iterator;
        return fill_array(array, base, step, array_length, iterator + 1);
    }}
    
    func main{{range_check_ptr: felt}}() {{
        alloc_locals;
        tempvar array_length = {num1};
        let (array: felt*) = alloc();
        fill_array(array, {num2}, {num3}, array_length, 0);
        assert_250_bit_element_array(array, array_length, 0);
        return ();
    }}
    
    "
    );

    // Create programs names and program
    let cairo_path_range_check = format!(
        "cairo_programs/range_check_{:?}.cairo",
        FUZZ_ITERATION_COUNT
    );
    let json_path_range_check =
        format!("cairo_programs/range_check_{:?}.json", FUZZ_ITERATION_COUNT);
    let _ = fs::write(&cairo_path_range_check, file_content.as_bytes());

    compile_program(&cairo_path_range_check, &json_path_range_check);

    let program_content_range_check = std::fs::read(&json_path_range_check).unwrap();

    // Run the program with default configurations
    let _ = cairo_run::cairo_run(
        &program_content_range_check,
        cairo_run_config,
        hint_executor,
    );

    // Remove files to save memory
    delete_files(&cairo_path_range_check, &json_path_range_check);
}

fn program_ec_op(
    num1: u128,
    cairo_run_config: &CairoRunConfig,
    hint_executor: &mut BuiltinHintProcessor,
) {
    let file_content = format!(
        "
    %builtins ec_op

    from starkware.cairo.common.cairo_builtins import EcOpBuiltin
    from starkware.cairo.common.ec_point import EcPoint
    from starkware.cairo.common.ec import recover_y

    func main{{ec_op_ptr: EcOpBuiltin*}}() {{
        let x = {:#02x};
        let r: EcPoint = recover_y(x);
        assert r.x = {:#02x};
        return ();
    }}
    
    ",
        num1, num1
    );

    // Create programs names and program
    let cairo_path_ec_op = format!("cairo_programs/ec_op_{:?}.cairo", FUZZ_ITERATION_COUNT);
    let json_path_ec_op = format!("cairo_programs/ec_op_{:?}.json", FUZZ_ITERATION_COUNT);
    let _ = fs::write(&cairo_path_ec_op, file_content.as_bytes());

    compile_program(&cairo_path_ec_op, &json_path_ec_op);

    let program_content_ec_op = std::fs::read(&json_path_ec_op).unwrap();

    // Run the program with default configurations
    let _ = cairo_run::cairo_run(&program_content_ec_op, cairo_run_config, hint_executor);

    // Remove files to save memory
    delete_files(&cairo_path_ec_op, &json_path_ec_op);
}

fn program_pedersen(
    num1: &Felt252,
    num2: &Felt252,
    cairo_run_config: &CairoRunConfig,
    hint_executor: &mut BuiltinHintProcessor,
) {
    let file_content = format!(
        "
    %builtins pedersen

    from starkware.cairo.common.cairo_builtins import HashBuiltin
    from starkware.cairo.common.hash import hash2
    
    func get_hash(hash_ptr: HashBuiltin*, num_a: felt, num_b: felt) -> (
        hash_ptr: HashBuiltin*, r: felt
    ) {{
        with hash_ptr {{
            let (result) = hash2(num_a, num_b);
        }}
        return (hash_ptr=hash_ptr, r=result);
    }}
    
    func builtins_wrapper{{
        pedersen_ptr: HashBuiltin*,
    }}(num_a: felt, num_b: felt) {{
        let (pedersen_ptr, result: felt) = get_hash(pedersen_ptr, num_a, num_b);
    
        return ();
    }}
    
    func builtins_wrapper_iter{{
        pedersen_ptr: HashBuiltin*,
    }}(num_a: felt, num_b: felt, n_iterations: felt) {{
        builtins_wrapper(num_a, num_b);
        if (n_iterations != 0) {{
            builtins_wrapper_iter(num_a, num_b, n_iterations - 1);
            tempvar pedersen_ptr = pedersen_ptr;
        }} else {{
            tempvar pedersen_ptr = pedersen_ptr;
        }}
    
        return ();
    }}
    
    func main{{
        pedersen_ptr: HashBuiltin*,
    }}() {{
        let num_a = {num1};
        let num_b = {num2};
        builtins_wrapper_iter(num_a, num_b, 50000);
    
        return ();
    }}
    "
    );

    // Create programs names and program
    let cairo_path_pedersen = format!("cairo_programs/pedersen_{:?}.cairo", FUZZ_ITERATION_COUNT);
    let json_path_pedersen = format!("cairo_programs/pedersen_{:?}.json", FUZZ_ITERATION_COUNT);
    let _ = fs::write(&cairo_path_pedersen, file_content.as_bytes());

    compile_program(&cairo_path_pedersen, &json_path_pedersen);

    let program_content_pedersen = std::fs::read(&json_path_pedersen).unwrap();

    // Run the program with default configurations
    let _ = cairo_run::cairo_run(&program_content_pedersen, cairo_run_config, hint_executor);

    // Remove files to save memory
    delete_files(&cairo_path_pedersen, &json_path_pedersen);
}

fn program_ecdsa(
    num1: &Felt252,
    num2: &Felt252,
    num3: &Felt252,
    num4: &Felt252,
    cairo_run_config: &CairoRunConfig,
    hint_executor: &mut BuiltinHintProcessor,
) {
    let file_content = format!(
        "
    %builtins ecdsa
    from starkware.cairo.common.serialize import serialize_word
    from starkware.cairo.common.cairo_builtins import SignatureBuiltin
    from starkware.cairo.common.signature import verify_ecdsa_signature
    
    func main{{ecdsa_ptr: SignatureBuiltin*}}() {{
        verify_ecdsa_signature(
            {num4},
            {num1},
            {num2},
            {num3},
        );
        return ();
    }}
    
    "
    );

    // Create programs names and program
    let cairo_path_ecdsa = format!("cairo_programs/ecdsa_{:?}.cairo", FUZZ_ITERATION_COUNT);
    let json_path_ecdsa = format!("cairo_programs/ecdsa_{:?}.json", FUZZ_ITERATION_COUNT);
    let _ = fs::write(&cairo_path_ecdsa, file_content.as_bytes());

    compile_program(&cairo_path_ecdsa, &json_path_ecdsa);

    let program_content_ecdsa = std::fs::read(&json_path_ecdsa).unwrap();

    // Run the program with default configurations
    let _ = cairo_run::cairo_run(&program_content_ecdsa, cairo_run_config, hint_executor);

    // Remove files to save memory
    delete_files(&cairo_path_ecdsa, &json_path_ecdsa);
}

fn compile_program(cairo_path: &str, json_path: &str) {
    let _output = Command::new("cairo-compile")
        .arg(cairo_path)
        .arg("--output")
        .arg(json_path)
        .output()
        .expect("failed to execute process");
}

fn delete_files(cairo_path: &str, json_path: &str) {
    fs::remove_file(cairo_path).expect("failed to remove file");
    fs::remove_file(json_path).expect("failed to remove file");
}
