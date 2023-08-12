use cairo_vm::{
    cairo_run::{cairo_run, CairoRunConfig},
    hint_processor::builtin_hint_processor::builtin_hint_processor_definition::BuiltinHintProcessor,
};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|data: (CairoRunConfig, &[u8])| {
            let (cairo_run_config, program_json) = data;
            let _ = cairo_run(
                program_json,
                &CairoRunConfig::default(),
                &mut BuiltinHintProcessor::new_empty(),
            );
            let _ = cairo_run(
                program_json,
                &cairo_run_config,
                &mut BuiltinHintProcessor::new_empty(),
            );
        });
    }
}
