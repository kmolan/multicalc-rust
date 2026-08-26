//! Draws an MJCF or URDF model in a Rerun viewer.
//!
//! Bodies are drawn at the pose the file states, which is each joint's own reference value.
//! Run: cargo run -p multicalc-robot-model --bin model_viewer -- <model-file>

use std::collections::HashSet;
use std::path::PathBuf;

use multicalc_robot_model::viewer::{self, ViewerOptions};
use rerun::RecordingStreamBuilder;

const USAGE: &str = "\
Usage: model_viewer <model-file> [options]

  --record <file.rrd>            write a recording instead of opening a viewer
  --geoms visual|collision|all   geom groups to draw (default: visual)
  --package-path <name>=<dir>    resolve package://<name>/... ; repeatable
  --frame-axes <metres>          frame axis length, 0 for none (default: 0.05)
  -h, --help                     this text";

/// The application id both modes record under.
const APPLICATION_ID: &str = "multicalc_model_viewer";

/// Exit code for a command line the parser could not read.
const USAGE_EXIT: u8 = 2;

fn main() -> std::process::ExitCode {
    let arguments = match Arguments::parse(std::env::args().skip(1)) {
        Ok(Some(arguments)) => arguments,
        // `--help`, which is a success.
        Ok(None) => {
            println!("{USAGE}");
            return std::process::ExitCode::SUCCESS;
        }
        Err(message) => {
            eprintln!("model_viewer: {message}\n\n{USAGE}");
            return std::process::ExitCode::from(USAGE_EXIT);
        }
    };
    match run(&arguments) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("model_viewer: {message}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run(arguments: &Arguments) -> Result<(), String> {
    let model = multicalc_robot_model::load_path(&arguments.file).map_err(|err| err.to_string())?;

    let builder = RecordingStreamBuilder::new(APPLICATION_ID);
    let stream = match &arguments.recording {
        Some(path) => builder.save(path.clone()),
        // As `RerunSink::live`: `RERUN_VIZ_URL` connects to that address, and under WSL (where the
        // virtualized GPU usually cannot launch a viewer) the default gRPC address reaches a
        // Windows-side viewer over shared localhost. Both need the viewer already running.
        None => match std::env::var("RERUN_VIZ_URL") {
            Ok(url) => builder.connect_grpc_opts(url),
            Err(_) if std::env::var_os("WSL_DISTRO_NAME").is_some() => builder.connect_grpc(),
            Err(_) => builder.spawn(),
        },
    }
    .map_err(|err| err.to_string())?;

    let report =
        viewer::log_model(&stream, &model, &arguments.options).map_err(|err| err.to_string())?;
    println!(
        "{} ({}): {} bodies, {} shapes drawn",
        model.name(),
        model.format(),
        model.body_count(),
        report.shapes()
    );

    let mut warned = HashSet::new();
    for file in report.skipped_meshes() {
        if warned.insert(file) {
            eprintln!("warning: mesh not drawn: {file}");
        }
    }

    stream.flush_blocking().map_err(|err| err.to_string())?;
    Ok(())
}

/// One command line.
struct Arguments {
    file: PathBuf,
    recording: Option<PathBuf>,
    options: ViewerOptions,
}

impl Arguments {
    /// Parses the command line, or `None` where it asks for the usage text.
    fn parse(arguments: impl Iterator<Item = String>) -> Result<Option<Self>, String> {
        let mut file: Option<PathBuf> = None;
        let mut recording = None;
        let mut options = ViewerOptions::new();

        let mut arguments = arguments;
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "-h" | "--help" => return Ok(None),
                "--record" => recording = Some(PathBuf::from(value(&argument, &mut arguments)?)),
                "--geoms" => {
                    options = options.with_groups(groups(&value(&argument, &mut arguments)?)?);
                }
                "--package-path" => {
                    let stated = value(&argument, &mut arguments)?;
                    let (name, directory) = stated.split_once('=').ok_or_else(|| {
                        format!("--package-path wants <name>=<dir>, got {stated}")
                    })?;
                    options = options.with_package_path(name.to_owned(), PathBuf::from(directory));
                }
                "--frame-axes" => {
                    let stated = value(&argument, &mut arguments)?;
                    let metres = stated
                        .parse::<f64>()
                        .map_err(|_| format!("--frame-axes wants a number, got {stated}"))?;
                    options = options.with_frame_axis_length(metres);
                }
                flag if flag.starts_with('-') => return Err(format!("unknown flag {flag}")),
                positional if file.is_none() => file = Some(PathBuf::from(positional)),
                extra => return Err(format!("only one model file is read, got {extra} too")),
            }
        }

        let file = file.ok_or_else(|| "no model file given".to_owned())?;
        Ok(Some(Arguments {
            file,
            recording,
            options,
        }))
    }
}

/// The value a flag takes, which it must be followed by.
fn value(flag: &str, arguments: &mut impl Iterator<Item = String>) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{flag} needs a value"))
}

/// The geom groups a `--geoms` choice names. `visual` is MuJoCo's own visible set.
fn groups(choice: &str) -> Result<Vec<u32>, String> {
    match choice {
        "visual" => Ok(vec![0, 1, 2]),
        "collision" => Ok(vec![3, 4]),
        "all" => Ok(vec![0, 1, 2, 3, 4, 5]),
        other => Err(format!(
            "--geoms wants visual, collision or all, got {other}"
        )),
    }
}
