use htm::CpuHTM2;
use htm_vis::visualise_cpu_htm2;
enum Scenario{
    S1,S2,S3
}
fn main() {
    match Scenario::S3{
        Scenario::S1 => {
            let mut htm = CpuHTM2::new(32,4);
            htm.add_globally_uniform_prob(16,4,2352);
            visualise_cpu_htm2(&htm, &[1,2,3,5,7], &[[2,4,4]], &[6,2,0], &[[2,2,4]])
        }
        Scenario::S2 => {
            let mut htm = CpuHTM2::new(32,4);
            htm.add_globally_uniform_prob(16,4,2352);
            visualise_cpu_htm2(&htm, &[1,2,3,5,7], &[[1,4,4],[1,4,4]], &[6,2,0], &[[1,2,4],[1,2,4]])
        }
        Scenario::S3 => {
            let mut htm = CpuHTM2::new(32,4);
            htm.add_globally_uniform_prob(16,4,2352);
            visualise_cpu_htm2(&htm, &[1,2,3,5,7], &[[4,4,2]], &[6,2,0], &[[2,2,4]])
        }
    }

}