use std::fs;

use anyhow::Context;
use cairo_lang_sierra::ProgramParser;
use cairo_lang_sierra_to_casm::compiler::CairoProgramWithContext;
use cairo_lang_sierra_to_casm::compiler::SierraToCasmConfig;
use cairo_lang_sierra_to_casm::metadata::calc_metadata;
use cairo_lang_utils::logging::init_logging;
use clap::Parser;
use indoc::indoc;

/// Compiles a Sierra file to CASM.
/// Exits with 0/1 if the compilation succeeds/fails.
#[derive(Parser, Debug)]
#[clap(version, verbatim_doc_comment)]
struct Args {
    /// The path of the file to compile.
    file: String,
    output: String,
}

fn main() -> anyhow::Result<()> {
    init_logging(log::LevelFilter::Off);
    log::info!("Starting Sierra compilation.");

    let args = Args::parse();

    let sierra_code = fs::read_to_string(args.file).with_context(|| "Could not read file!")?;
    let Ok(program) = ProgramParser::new().parse(&sierra_code) else {
        anyhow::bail!(indoc! {"
            Failed to parse sierra program.
            Note: StarkNet contracts should be compiled with `starknet-sierra-compile`."
        })
    };
    let metadata = &calc_metadata(&program, Default::default())
        .with_context(|| "Failed calculating Sierra variables.")?;
    let cairo_program = cairo_lang_sierra_to_casm::compiler::compile(
        &program,
        metadata,
        SierraToCasmConfig { gas_usage_check: true, max_bytecode_size: usize::MAX },
    )
    .with_context(|| "Compilation failed.")?;
    let cairo_program_with_context =
        CairoProgramWithContext::new(&cairo_program, &program, metadata);
    let json_result = serde_json::to_string(&cairo_program_with_context);
    match json_result {
        Ok(json_string) => {
            fs::write(&args.output, &json_string)
                .with_context(|| format!("Failed to write output to {}", args.output))?;
        }
        Err(e) => {
            anyhow::bail!("Failed to serialize CairoProgram: {}", e);
        }
    }
    Ok(())
    // fs::write(args.output, format!("{cairo_program}")).with_context(|| "Failed to write output.")
}
