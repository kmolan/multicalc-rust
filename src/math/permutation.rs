use crate::math::factorial as factorial;

pub fn get(num_objects: u64, num_selected: u64) -> u64
{
    return factorial::get(num_objects)/factorial::get(num_objects - num_selected);
}