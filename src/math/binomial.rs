use crate::math::factorial as factorial;

pub fn get(num_objects: u64, num_selected: u64) -> u64
{
    return factorial::get_with_range(num_objects, num_objects - num_selected)/factorial::get(num_selected);
}