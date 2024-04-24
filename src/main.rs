use clap::{Arg, Command};
use plotters::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
#[macro_use]
extern crate rustacuda;
extern crate rustacuda_core;

use rustacuda::prelude::*;
use rustacuda::memory::{DeviceBuffer, DeviceBox};
use std::ffi::CString;

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
        grid[scale][scale + i] = (scale - i) as f64 / scale as f64;
        grid[scale + i][scale] = (scale - i) as f64 / scale as f64;
    }
    grid
}

fn relax_potential(grid: &mut [&mut [f64]], scale: usize, epsilon: f64) -> Result<(), Box<dyn Error>> {
    rustacuda::init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let module_data = CString::new(include_str!("../relaxKernel.ptx"))?;
    let module = Module::load_from_string(&module_data)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let width = 2 * scale;
    let size = width * width;

    // Flatten the grid for GPU processing
    let mut flat_grid: Vec<f64> = Vec::with_capacity(size);
    for row in grid.iter() {
        flat_grid.extend_from_slice(row);
    }

    let mut grid_dev = std::pin::pin!(DeviceBuffer::from_slice(&flat_grid)?);
    let mut converged = DeviceBox::new(&false)?;
    let mut host_grid = vec![0.0f64; size];

    unsafe {
        let blocks = (width as u32 + 15) / 16;
        let threads = 16;
        let mut iterations = 0;
        let mut pinned = std::pin::pin!(converged);

        loop {
            pinned.as_mut().set(DeviceBox::new(&true)?);

            launch!(module.relaxKernel<<<(blocks, blocks, 1), (threads, threads, 1), 0, stream>>>(
                grid_dev.as_device_ptr(),
                width as i32,
                scale as i32,
                epsilon,
                pinned.as_device_ptr(),
                size as i32  // Total number of elements
            ))?;

            stream.synchronize()?;

            let mut host_converged = false;
            pinned.copy_to(&mut host_converged)?;

            if host_converged {
                break;
            }

            iterations += 1;
            if iterations > 100000 {
                println!("Failed to converge after 100000 iterations.");
                break;
            }
        }
    }

    // Copy the result back to the host
    grid_dev.copy_to(&mut host_grid)?;

    // Unflatten the grid back to 2D
    for (i, row) in grid.iter_mut().enumerate() {
        let start = i * width;
        let end = start + width;
        row.copy_from_slice(&host_grid[start..end]);
    }

    println!("Grid relaxed.");

    Ok(())
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
