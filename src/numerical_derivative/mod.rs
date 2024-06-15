pub mod mode;
pub mod single_derivative;
pub mod double_derivative;
pub mod triple_derivative;

#[cfg(feature = "std")]
pub mod jacobian;

pub mod hessian;

#[cfg(test)]
mod test;