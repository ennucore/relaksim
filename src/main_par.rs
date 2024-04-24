use clap::{Command, Arg};
use std::io::{Write, BufWriter};
use plotters::prelude::*;
use std::error::Error;
use rayon::prelude::*;
use ndarray::{Array2, ArrayView2, arr2, Dim, Zip};
use std::sync::atomic::{AtomicBool, Ordering};
use ndarray::parallel::prelude::*;
// import mutex and arc
use std::sync::{Mutex, Arc};


fn main() -> Result<(), Box<dyn Error>> {
    let matches = Command::new("Laplace Solver")
        .version("1.0")
        .author("Lev Chizhov")
        .about("Solves Laplace's equation for an L-shaped duct using the relaxation method.")
        .arg(Arg::new("num")
             .short('n')
             .long("num")
             .value_name("NUM")
             .help("Sets the scale factor for grid size")
             .default_value("100"))
        .get_matches();

    let n: usize = matches.get_one::<String>("num").unwrap().parse().unwrap();
    let scale: usize = n / 2;

    let mut grid = initialize_boundary_conditions(scale);

    let epsilon = 1e-5;
    relax_potential(&mut grid, scale, epsilon);

    plot_results(grid)?;
    Ok(())
}



fn initialize_boundary_conditions(scale: usize) -> Array2<f64> {
    let mut grid = Array2::<f64>::zeros((2 * scale, 2 * scale));
    for i in 0..scale {
        grid[(scale, i)] = i as f64 / scale as f64;
        grid[(scale + i, scale)] = (scale - i) as f64 / scale as f64;
    }
    grid
}

fn relax_potential(grid: &mut Array2<f64>, scale: usize, epsilon: f64) {
    let shape = grid.dim();
    let mut converged = false;
    while !converged {
        let converged_ = Arc::new(Mutex::new(true));
        let mut new_grid = grid.clone();
        
        new_grid.axis_iter_mut(ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                if i == 0 || i == shape.0 - 1 { return; } // Skip the boundary rows
                row.iter_mut()
                    .enumerate()
                    .for_each(|(j, v)| {
                        if j == 0 || j == shape.1 - 1 { return; } // Skip the boundary columns
                        if i < scale || j < scale { // Skip fixed potential points
                            let new_v = (
                                grid[(i + 1, j)] +
                                grid[(i - 1, j)] +
                                grid[(i, j + 1)] +
                                grid[(i, j - 1)]
                            ) / 4.;
                            if (new_v - *v).abs() > epsilon && *converged_.lock().unwrap() {
                                *(converged_.lock().unwrap()) = false;
                            }
                            *v = new_v;
                        }
                    });
            });

        *grid = new_grid;
        converged = *converged_.lock().unwrap();
    }
}


// fn relax_potential(grid: &mut Array2<f64>, scale: usize, epsilon: f64) {
//     let shape = grid.dim();
//     let converged = AtomicBool::new(false);

//     while !converged.load(Ordering::SeqCst) {
//         converged.store(true, Ordering::SeqCst);

//         // Perform the update using a mutable view to ensure all data access goes through grid_ref
//         let mut grid_ref = grid.view_mut();

//         // Use Zip to apply updates in place
//         Zip::indexed(&mut grid_ref).par_for_each(|(i, j), v| {
//             if i == 0 || i == shape.0 - 1 || j == 0 || j == shape.1 - 1 {
//                 // Skip the boundary cells
//                 return;
//             }

//             // Use only grid_ref to access grid elements, resolving borrowing conflicts
//             if i < scale || j < scale {
//                 let old_v = *v;
//                 let new_v = (
//                     grid_ref[(i + 1, j)] +
//                     grid_ref[(i - 1, j)] +
//                     grid_ref[(i, j + 1)] +
//                     grid_ref[(i, j - 1)]
//                 ) / 4.0;

//                 if (new_v - old_v).abs() > epsilon {
//                     converged.store(false, Ordering::SeqCst);
//                     *v = new_v;
//                 }
//             }
//         });
//     }
// }

// use convolutions_rs::convolutions::*;
// use convolutions_rs::Padding;
// use ndarray::*;

// fn relax_potential(grid: &mut Array2<f64>, scale: usize, epsilon: f64) {
//     // Create a kernel that averages the four adjacent cells
//     let kernel: Array4<f64> = Array::from_shape_vec(
//         (1, 1, 3, 3),
//         vec![
//             0.0, 0.25, 0.0,
//             0.25, 0.0, 0.25,
//             0.0, 0.25, 0.0,
//         ]
//     )
//     .unwrap();

//     let mut converged = false;

//     // Mask to determine which points can be updated
//     let mut update_mask = Array::from_elem(grid.dim(), 1.);
//     // Assume the boundaries are at the first and last rows/columns
//     update_mask.slice_mut(s![0, ..]).fill(0.);
//     update_mask.slice_mut(s![-1, ..]).fill(0.);
//     update_mask.slice_mut(s![.., 0]).fill(0.);
//     update_mask.slice_mut(s![.., -1]).fill(0.);
//     // Exclude fixed potential points
//     // Example: Exclude the points based on some condition related to `scale`
//     for i in 0..update_mask.nrows() {
//         for j in 0..update_mask.ncols() {
//             if !(i < scale || j < scale) { // Example condition
//                 update_mask[(i, j)] = 0.;
//             }
//         }
//     }

//     while !converged {
//         let input = grid.clone().insert_axis(Axis(0));  // Convert to (1, height, width) format
//         let conv_layer = ConvolutionLayer::new(kernel.clone(), None, 1, Padding::Same);  // Use same padding
//         let new_grid_ = conv_layer.convolve(&input).remove_axis(Axis(0));
//         let diff = (new_grid_ - grid.view()) * &update_mask;
//         *grid += &diff;

//         converged = !diff.par_iter().any(|&x| x.abs() > epsilon);
//     }
// }



fn plot_results(grid: Array2<f64>) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("plot.png", (600, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    // Grid dimensions
    let (height, width) = grid.dim();

    for y in 0..height {
        for x in 0..width {
            let val = grid[(y, x)];
            let color = HSLColor(0.7 - 0.7 * val, 0.7, 0.5).to_rgba();
            root.draw(&Pixel::new((x as i32, y as i32), color))?;
        }
    }

    root.present()?;
    println!("Plot is saved as 'plot.png'.");
    Ok(())
}

