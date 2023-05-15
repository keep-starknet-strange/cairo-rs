use cairo_vm::{
    felt::Felt252,
    types::program::Program,
    vm::{runners::cairo_runner::CairoRunner, vm_core::VirtualMachine},
};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use num_traits::Zero;

#[cfg(feature = "with_mimalloc")]
use mimalloc::MiMalloc;

#[cfg(feature = "with_mimalloc")]
#[global_allocator]
static ALLOC: MiMalloc = MiMalloc;

fn parse_program(c: &mut Criterion) {
    //Picked the biggest one at the time of writing
    let program = include_bytes!("../cairo_programs/benchmarks/keccak_integration_benchmark.json");
    c.bench_function("parse program", |b| {
        b.iter(|| {
            _ = Program::from_bytes(black_box(program.as_slice()), black_box(Some("main")))
                .unwrap();
        })
    });
}

fn build_many_runners(c: &mut Criterion) {
    //Picked the biggest one at the time of writing
    let program = include_bytes!("../cairo_programs/benchmarks/keccak_integration_benchmark.json");
    let program = Program::from_bytes(program.as_slice(), Some("main")).unwrap();
    c.bench_function("build runner", |b| {
        b.iter(|| {
            _ = black_box(
                CairoRunner::new(
                    black_box(&program),
                    black_box("starknet_with_keccak"),
                    black_box(false),
                )
                .unwrap(),
            );
        })
    });
}

fn load_program_data(c: &mut Criterion) {
    //Picked the biggest one at the time of writing
    let program = include_bytes!("../cairo_programs/benchmarks/keccak_integration_benchmark.json");
    let program = Program::from_bytes(program.as_slice(), Some("main")).unwrap();
    c.bench_function("initialize", |b| {
        b.iter_batched(
            || {
                (
                    CairoRunner::new(&program, "starknet_with_keccak", false).unwrap(),
                    VirtualMachine::new(false),
                )
            },
            |(mut runner, mut vm)| _ = black_box(runner.initialize(black_box(&mut vm)).unwrap()),
            BatchSize::SmallInput,
        )
    });
}

fn add_u64_with_felt252(c: &mut Criterion) {
    // There are 9 possible cases:
    // - The felt is `0`
    // - The felt is `-1` and the result in range
    // - The felt is `-1` but the result is out of range (the u64 is `0`)
    // - The felt is positive and the result in range
    // - The felt is positive and the result is out of range
    // - The felt is `> u64::MAX` which always causes to be out of range
    // - The felt is `<= -2` and the result is in range
    // - The felt is `<= -2` and the result is out of range
    // - The felt is `< -u64::MAX` which always causes to be out of range
    // I consider all of these cases because branching is the most likely
    // bottleneck here.
    let cases = [
        (1u64, Felt252::zero()),
        (1u64, Felt252::from(-1i128)),
        (0u64, Felt252::from(-1i128)),
        (0u64, Felt252::from(1i128)),
        (1u64, Felt252::from(u64::MAX as i128)),
        (0u64, Felt252::from(u64::MAX as i128 + 1i128)),
        (2u64, Felt252::from(-2i128)),
        (1u64, Felt252::from(-2i128)),
        (0u64, Felt252::from(-(u64::MAX as i128) - 1i128)),
    ];
    let mut group = c.benchmark_group("add_u64_with_felt");
    for (i, case) in cases.iter().enumerate() {
        group.bench_with_input(BenchmarkId::from_parameter(i), case, |b, (lhs, rhs)| {
            b.iter(|| *lhs + rhs);
        });
    }
    group.finish();
}

criterion_group!(felt, add_u64_with_felt252);
criterion_group!(runner, build_many_runners, load_program_data, parse_program);
criterion_main!(felt, runner);
