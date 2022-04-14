pub fn shape1<T, const DIM0: usize>(_mat: &[T; DIM0]) -> [usize; 1] {
    [DIM0]
}

pub fn shape2<T, const DIM0: usize, const DIM1: usize>(_mat: &[[T; DIM0]; DIM1]) -> [usize; 2] {
    [DIM0, DIM1]
}

pub fn shape3<T, const DIM0: usize, const DIM1: usize, const DIM2: usize>(_mat: &[[[T; DIM0]; DIM1]; DIM2]) -> [usize; 3] {
    [DIM0, DIM1, DIM2]
}

pub fn shape4<T, const DIM0: usize, const DIM1: usize, const DIM2: usize, const DIM3: usize>(_mat: &[[[[T; DIM0]; DIM1]; DIM2]; DIM3]) -> [usize; 4] {
    [DIM0, DIM1, DIM2, DIM3]
}

pub fn shape5<T, const DIM0: usize, const DIM1: usize, const DIM2: usize, const DIM3: usize, const DIM4: usize>(_mat: &[[[[[T; DIM0]; DIM1]; DIM2]; DIM3]; DIM4]) -> [usize; 5] {
    [DIM0, DIM1, DIM2, DIM3, DIM4]
}

pub fn col_vec<T, const DIM0: usize>(mat: [T; DIM0]) -> [[T; 1]; DIM0] {
    unsafe { std::mem::transmute(mat) }
}

pub fn row_vec<T, const DIM0: usize>(mat: [T; DIM0]) -> [[T; DIM0]; 1] {
    unsafe { std::mem::transmute(mat) }
}

pub fn unsqueeze2_1<T, const DIM0: usize>(mat: [T; DIM0]) -> [[T; 1]; DIM0] {
    col_vec(mat)
}

pub fn unsqueeze2_0<T, const DIM0: usize>(mat: [T; DIM0]) -> [[T; DIM0]; 1] {
    row_vec(mat)
}

pub fn squeeze2_1<T, const DIM1: usize>(mat: [[T; 1]; DIM1]) -> [T; DIM1] {
    unsafe { std::mem::transmute(mat) }
}

pub fn squeeze2_0<T, const DIM0: usize>(mat: [[T; DIM0]; 1]) -> [T; DIM0] {
    unsafe { std::mem::transmute(mat) }
}

pub fn unsqueeze3_0<T, const DIM0: usize, const DIM1: usize>(mat: [[T; DIM0]; DIM1]) -> [[[T; DIM0]; DIM1]; 1] {
    unsafe { std::mem::transmute(mat) }
}

pub fn unsqueeze3_1<T, const DIM0: usize, const DIM1: usize>(mat: [[T; DIM0]; DIM1]) -> [[[T; DIM0]; 1]; DIM1] {
    unsafe { std::mem::transmute(mat) }
}

pub fn unsqueeze3_2<T, const DIM0: usize, const DIM1: usize>(mat: [[T; DIM0]; DIM1]) -> [[[T; 1]; DIM0]; DIM1] {
    unsafe { std::mem::transmute(mat) }
}


pub fn squeeze3_0<T, const DIM0: usize, const DIM1: usize>(mat: [[[T; DIM0]; DIM1]; 1]) -> [[T; DIM0]; DIM1] {
    unsafe { std::mem::transmute(mat) }
}

pub fn squeeze3_1<T, const DIM0: usize, const DIM2: usize>(mat: [[[T; DIM0]; 1]; DIM2]) -> [[T; DIM0]; DIM2] {
    unsafe { std::mem::transmute(mat) }
}

pub fn squeeze3_2<T, const DIM2: usize, const DIM1: usize>(mat: [[[T; 1]; DIM1]; DIM2]) -> [[T; DIM1]; DIM2] {
    unsafe { std::mem::transmute(mat) }
}
