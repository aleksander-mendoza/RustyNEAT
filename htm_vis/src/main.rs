// use htm::CpuHTM;
// use htm_vis::visualise_cpu_htm2;
// enum Scenario{
//     S1,S2,S3
// }
// fn main() {
//     let mut r = rand::thread_rng();
//     match Scenario::S3{
//         Scenario::S1 => {
//             let mut htm = CpuHTM::new(32, 4);
//             htm.add_globally_uniform_prob(16,4,&mut r);
//             visualise_cpu_htm2(&htm,  &[[2,4,4]],  &[[2,2,4]],&[1,2,3,5,7],&[6,2,0],0.2,0.2)
//         }
//         Scenario::S2 => {
//             let mut htm = CpuHTM::new(32, 4);
//             htm.add_globally_uniform_prob(16,4,&mut r);
//             visualise_cpu_htm2(&htm, &[[1,4,4],[1,4,4]], &[[1,2,4],[1,2,4]],&[1,2,3,5,7],&[6,2,0],0.2,0.2)
//         }
//         Scenario::S3 => {
//             let mut htm = CpuHTM::new(32, 4);
//             htm.add_globally_uniform_prob(16,4,&mut r);
//             visualise_cpu_htm2(&htm, &[[4,4,2]], &[[2,2,4]],&[1,2,3,5,7],&[6,2,0],0.2,0.2)
//         }
//     }
//
// }