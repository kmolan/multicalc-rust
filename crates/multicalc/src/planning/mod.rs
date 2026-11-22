//! Path planners: search over a grid map, or over a continuous state space by sampling.
//!
//! - [`GridPlanner`] — Dijkstra, A\*, weighted A\* and Theta\* over an
//!   [`OccupancyMap`](crate::mapping::OccupancyMap), into a [`GridSearchWorkspace`].
//! - [`TraversalCost`] — what entering a cell costs, with [`UniformCost`] over a plain map and
//!   [`CostmapCost`] over an inflation [`CostGrid`](crate::mapping::CostGrid).
//! - [`StateSpace`] and [`StateValidity`] — where states live and which of them are free, with
//!   [`BoxSpace`] over a box in Euclidean space.
//! - [`Rrt`], [`RrtStar`] and [`Prm`] — sampling planners over that space, into an
//!   [`RrtWorkspace`] or a [`PrmWorkspace`].
//! - [`PlanReport`] — the plan, its cost, and what the search spent finding it.
//!
//! **Planning is a planning-time cost, never work for a 1 kHz control loop.** A search runs when
//! the goal or the map changes; the loop consumes the [`PolylinePath`](crate::motion::PolylinePath)
//! that comes back, through [`MinimumSnapPlanner`](crate::motion::MinimumSnapPlanner) or
//! [`pure_pursuit_curvature`](crate::control::pure_pursuit_curvature).
//!
//! Every search runs in a caller-owned workspace, so nothing here allocates and the memory is
//! sized and placed by the caller — a `static`, or a `Box` where `alloc` is available.

mod frontier;
mod grid_planner;
mod grid_workspace;
mod plan_report;
mod prm;
mod rrt;
mod sampling;
mod state_space;
mod traversal_cost;

pub use grid_planner::{GridConnectivity, GridHeuristic, GridPlanner, GridSearch};
pub use grid_workspace::GridSearchWorkspace;
pub use plan_report::PlanReport;
pub use prm::{Prm, PrmWorkspace};
pub use rrt::{Rrt, RrtStar, RrtWorkspace};
pub use state_space::{BoxSpace, StateSpace, StateValidity};
pub use traversal_cost::{CostmapCost, TraversalCost, UniformCost};
