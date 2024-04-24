use clap::{Arg, Command};
use plotters::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() -> Result<(), Box<dyn Error>> {
    let matches = Command::new("Laplace Solver")
        .version("1.0")
        .author("Lev Chizhov")
        .about("Solves Laplace's equation for an L-shaped duct using the relaxation method.")
        .arg(
            Arg::new("scale")
                .short('s')
                .long("scale")
                .value_name("SCALE")
                .help("Sets the scale factor for grid size")
                .default_value("50"),
        )
        .get_matches();

    let scale: usize = matches.get_one::<String>("scale").unwrap().parse().unwrap();

    let mut grid_vec = initialize_boundary_conditions(scale);
    // let mut grid = grid_vec.iter_mut().map(|v| v.as_mut_slice()).collect::<Vec<&mut [f64]>>().as_mut_slice();
    let mut binding = grid_vec
        .iter_mut()
        .map(|v| v.as_mut_slice())
        .collect::<Vec<&mut [f64]>>();
    let mut grid = binding.as_mut_slice();

    let epsilon = 1e-5;
    relax_potential(&mut grid, scale, epsilon);

    plot_results(&grid_vec)?;
    Ok(())
}

fn initialize_boundary_conditions(scale: usize) -> Vec<Vec<f64>> {
    let mut grid = vec![vec![0.0; 2 * scale]; 2 * scale];

    for i in 0..scale {
        grid[scale][i] = i as f64 / scale as f64;
        grid[scale + i][scale] = (scale - i) as f64 / scale as f64;
    }
    grid
}

fn relax_potential(grid: &mut [&mut [f64]], scale: usize, epsilon: f64) {
    let mut converged = false;
    while !converged {
        converged = true;
        let mut last_row_cached = grid[0].to_vec();
        let mut last_value_cached = grid[1][0];
        for i in 1..2 * scale - 1 {
            let last_row_cached_ = grid[i].to_vec();
            for j in 1..2 * scale - 1 {
                if i < scale || j > scale {
                    let new_v = 0.25
                        * (grid[i + 1][j]
                            + last_row_cached[j]
                            + grid[i][j + 1]
                            + last_value_cached);
                    if (new_v - grid[i][j]).abs() > epsilon {
                        converged = false;
                    }
                    grid[i][j] = new_v;
                }
                last_value_cached = grid[i][j];
            }
            last_row_cached = last_row_cached_;
            last_value_cached = grid[i + 1][0];
        }
    }
}

fn export_grid(grid: &Vec<Vec<f64>>, file_path: &str) {
    let file = File::create(file_path).unwrap();
    let mut writer = BufWriter::new(file);
    for row in grid {
        let line = row
            .iter()
            .map(|&v| v.to_string())
            .collect::<Vec<String>>()
            .join(",");
        writeln!(writer, "{}", line).unwrap();
    }
}

fn plot_results(grid: &[Vec<f64>]) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("plot.png", (600, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    for (y, row) in grid.iter().enumerate() {
        for (x, &val) in row.iter().enumerate() {
            let color = HSLColor(0.7 - 0.7 * val, 0.7, 0.5).to_rgba();
            root.draw(&Pixel::new((x as i32, y as i32), color))?;
        }
    }

    root.present()?;
    println!("Plot is saved as 'plot.png'.");
    Ok(())
}
