use std::sync::atomic::{AtomicUsize, Ordering};

pub fn parallel_map_collect<A, T, B>(a: &[A], t: &mut [T], f: impl Fn(&A, &mut T) -> B + Send + Sync) -> Vec<B> {
    let mut b: Vec<B> = Vec::with_capacity(a.len());
    unsafe { b.set_len(a.len()) }
    parallel_map_vector(a, t, &mut b, |a, t, o| {
        let o = o as *mut B;
        unsafe { o.write(f(a, t)) }
    });
    b
}
/**Maps every element from A to B. Every thread has mutable access to its own element T*/
pub fn parallel_map_vector<A, T, B, F>(a: &[A], t: &mut [T], b: &mut [B], f: F) where F:Fn(&A, &mut T, &mut B) + Send + Sync{
    assert_eq!(a.len(), b.len());
    let atomic_counter = AtomicUsize::new(0);
    let b_ptr: *mut B = b.as_mut_ptr();
    let b_ptr = b_ptr as usize;
    let b_len = b.len();

    let a_ptr: *const A = a.as_ptr();
    let a_ptr = a_ptr as usize;
    let a_len = a.len();

    let th: Vec<std::thread::JoinHandle<()>> = t.iter_mut().map(|target| {
        let t_ptr = target as *mut T;
        let t_ptr_ = t_ptr as usize;
        let f_ref = &f;
        let f_ptr = f_ref as *const F;
        let f_ptr = f_ptr as usize;
        let ac_ref = &atomic_counter;
        let ac_ptr = ac_ref as *const AtomicUsize;
        let ac_ptr = ac_ptr as usize;
        std::thread::spawn(move || {
            let a = unsafe { std::slice::from_raw_parts(a_ptr as *const A, a_len) };
            let t_ptr = t_ptr_ as *mut T;
            let target = unsafe { &mut *t_ptr };
            let f = f_ptr as *const F;
            let f = unsafe{&*f};
            let ac_ref= ac_ptr as *const AtomicUsize;
            let ac_ref = unsafe{&*ac_ref};
            loop {
                let idx = ac_ref.fetch_add(1, Ordering::Relaxed);
                if idx < a_len {
                    let b = unsafe { std::slice::from_raw_parts_mut(b_ptr as *mut B, b_len) };
                    f(&a[idx], target, &mut b[idx])
                } else {
                    break;
                }
            }
        })
    }).collect();
    th.into_iter().for_each(|t|t.join().unwrap())
}


#[cfg(test)]
mod tests {
    use crate::parallel::{parallel_map_vector, parallel_map_collect};

    #[test]
    fn test() {
        let a: Vec<i32> = (0..1024).collect();
        let mut b = vec![0; 1024];
        let mut t = vec![0, 0, 0];
        parallel_map_vector(&a, &mut t, &mut b, |a, t, o| {
            *o = a + 10;
            *t += 1;
        });
        for (i, &b) in b.iter().enumerate() {
            assert_eq!(b, i as i32 + 10)
        }
        for &tt in &t {
            assert_ne!(tt, 0, "{:?}", t);
        }
    }

    #[test]
    fn test2() {
        let a: Vec<i32> = (0..1024).collect();
        let mut t = vec![0, 0, 0];
        let b = parallel_map_collect(&a, &mut t, |a, t| {
            *t += 1;
            a + 10
        });
        for (i, &b) in b.iter().enumerate() {
            assert_eq!(b, i as i32 + 10)
        }
        for &tt in &t {
            assert_ne!(tt, 0, "{:?}", t);
        }
    }
}