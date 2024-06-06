pub fn get(num: u64) -> u64
{
    if num < 2 
    {   
        return 1;
    }
    if num == 2
    {
        return 2;
    }

    return (num)*get(num-1);
}

//used for binomial coefficient calculation
// n*(n-1)*(n-2)*....*(num - stop - 1)
pub fn get_with_range(num: u64, stop:u64) -> u64
{
    if num < 2 
    {   
        return 1;
    }
    if num == 2
    {
        return 2;
    }

    if num == stop
    {
        return 1;
    }

    return (num)*get_with_range(num-1, stop);
}