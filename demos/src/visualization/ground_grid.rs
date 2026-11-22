//! A floor grid for the 3D demos to sit on.

/// A CHROME `n×n`-cell unit grid on the z = 0 plane, centered at the origin, as 2-point strips.
/// Log once at tick 0 with
/// `line_strips3d("world/ground", &ground_grid(3.0, 1.0), &[CHROME], &[0.004])`.
#[must_use]
pub fn ground_grid(half_extent: f64, step: f64) -> Vec<Vec<[f64; 3]>> {
    let mut strips = Vec::new();
    let n = (half_extent / step) as i64;
    for k in -n..=n {
        let coord = k as f64 * step;
        strips.push(vec![[coord, -half_extent, 0.0], [coord, half_extent, 0.0]]);
        strips.push(vec![[-half_extent, coord, 0.0], [half_extent, coord, 0.0]]);
    }
    strips
}
