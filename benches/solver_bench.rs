use std::hint::black_box;
use std::time::Instant;

use fluid_sim_2d::sim::forces::SimCommand;
use fluid_sim_2d::sim::{FluidSolver, GridSize, SimulationState};
use fluid_sim_2d::SimulationConfig;
use glam::Vec2;

fn main() {
    println!("fluid-sim-2d manual benchmark");
    println!("build profile: release-like benchmark harness");

    run_solver_benchmark("solver_step_160x90", SimulationConfig::default(), 240);

    run_solver_benchmark(
        "solver_step_256x144",
        SimulationConfig {
            grid_width: 256,
            grid_height: 144,
            solver_iterations: 50,
            ..SimulationConfig::default()
        },
        120,
    );
}

fn run_solver_benchmark(name: &str, config: SimulationConfig, steps: usize) {
    let grid = GridSize::new(config.grid_width, config.grid_height, config.cell_size)
        .expect("benchmark grid should be valid");
    let mut solver = FluidSolver::new(config.clone());
    let mut state = SimulationState::new(grid);
    let commands = build_benchmark_commands(grid);

    for _ in 0..20 {
        solver.step(&mut state, &commands);
    }

    state.clear();

    let start = Instant::now();
    for step in 0..steps {
        let active_commands = if step % 8 == 0 { &commands[..] } else { &[] };
        solver.step(black_box(&mut state), black_box(active_commands));
    }
    let elapsed = start.elapsed();

    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let avg_ms = total_ms / steps as f64;
    let steps_per_second = steps as f64 / elapsed.as_secs_f64();

    println!(
        "{name}: grid={}x{} iterations={} steps={} total={total_ms:.2}ms avg={avg_ms:.3}ms steps/s={steps_per_second:.1}",
        config.grid_width,
        config.grid_height,
        config.solver_iterations,
        steps,
    );
}

fn build_benchmark_commands(grid: GridSize) -> [SimCommand; 2] {
    let center = Vec2::new(
        grid.nx as f32 * grid.cell_size * 0.5,
        grid.ny as f32 * grid.cell_size * 0.5,
    );

    [
        SimCommand::AddForce {
            position: center,
            delta: Vec2::new(2.5, 1.0),
            radius: 8.0,
        },
        SimCommand::AddDye {
            position: center,
            amount: 1.5,
            radius: 8.0,
        },
    ]
}
