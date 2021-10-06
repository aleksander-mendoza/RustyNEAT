use ocl::{ProQue, SpatialDims, flags, Platform, Device, Buffer, Program, Kernel, Queue};
use ndalgebra::lin_alg_program::LinAlgProgram;
use ndalgebra::mat::{Mat, MatError, AsShape};
use crate::context::NeatContext;
use ndalgebra::kernel_builder::KernelBuilder;
use ocl::core::Error;

pub struct Evol {
    lin_alg: LinAlgProgram,
    program: Program,
    lidar_count: usize,
    width:usize,
    height:usize
}
pub const AGENT_ATTRIBUTES:usize = 6;
pub const LIDAR_ATTRIBUTES:usize = 2;


impl Evol {

    pub fn new(width:usize,height:usize,
               hunger_change_per_step:f32,
               lidar_angles:&[f32],
               steps:usize, step_len:f32,
               context:&NeatContext) -> Result<Self, String> {
        if width==0||height==0{
            return Err(format!("Width and height must be non-zero"));
        }
        let central_lidar_idx = lidar_angles.iter().position(|&l|l==0f32).ok_or_else(||format!("One of the lidar angles must be zero, because it will tell how much the agent can advance forward"))?;

        let mut src = format!("\
__kernel void evol(__global uchar * borders,
                   __global float * agents,
                   __global float * lidars,
                   size_t borders_row_stride,
                   size_t borders_col_stride,
                   size_t agents_row_stride,
                   size_t agents_col_stride,
                   size_t lidars_row_stride,
                   size_t lidars_col_stride,
                   size_t lidars_depth_stride){{
    float lidar_angles[] = {{{}}};
    uchar lidars_len = {lidars_len};
    uchar steps = {steps};
    float step_len = {step_len};
    float max_lidar_dist = steps*step_len;
    float hunger_change_per_step = {hunger_change_per_step};
    size_t agent_idx = get_global_id(0);
    size_t agent_offset = agent_idx*agents_row_stride;
    size_t agent_x_offset = agent_offset+agents_col_stride*0;
    size_t agent_y_offset = agent_offset+agents_col_stride*1;
    size_t agent_angle_offset = agent_offset+agents_col_stride*2;
    size_t agent_hunger_offset = agent_offset+agents_col_stride*3;
    size_t agent_rotation_action_offset = agent_offset+agents_col_stride*4;
    size_t agent_movement_action_offset = agent_offset+agents_col_stride*5;
    float x = agents[agent_x_offset];
    float y = agents[agent_y_offset];
    float angle = agents[agent_angle_offset];
    float movement_action = agents[agent_movement_action_offset];
    size_t lidar_offset = agent_idx*lidars_row_stride;
    float prev_central_lidar_distance = lidars[lidar_offset+{central_lidar_idx}*lidars_col_stride+lidars_depth_stride*0];
    float position_delta = clamp((float)movement_action, (float)0.0, (float)prev_central_lidar_distance);
    x = x + sin(angle) * position_delta;
    y = y + cos(angle) * position_delta;
    agents[agent_x_offset] = x;
    agents[agent_y_offset] = y;
    float rotation_action = agents[agent_rotation_action_offset];

    angle+=rotation_action;
    agents[agent_angle_offset] = angle;

    for(uchar i=0;i<lidars_len;i++){{
        float lidar_angle = angle + lidar_angles[i];
        float lidar_sin = sin(lidar_angle);
        float lidar_cos = cos(lidar_angle);
        float dist = 0;
        ushort seen_food = 0;
        while(true){{
            float new_dist = dist + step_len;
            float lidar_x = x + lidar_sin * new_dist;
            float lidar_y = y + lidar_cos * new_dist;
            if(lidar_x<0||lidar_y<0||lidar_x>={width}||lidar_y>={height}){{
                break;
            }}
            size_t idx = ((size_t)lidar_y)*borders_row_stride+((size_t)lidar_x)*borders_col_stride;
            uchar pixel = borders[idx];
            if(pixel==255 || new_dist >= max_lidar_dist){{
                break;
            }}
            seen_food += pixel;
            dist = new_dist;
        }}
        size_t lidar_offset2 = lidar_offset+i*lidars_col_stride;
        size_t lidar_dist_offset = lidar_offset2+lidars_depth_stride*0;
        size_t lidar_seen_food_offset = lidar_offset2+lidars_depth_stride*1;
        lidars[lidar_dist_offset] = dist;
        lidars[lidar_seen_food_offset] = seen_food;
    }}
    size_t borders_offset = y*borders_row_stride+x*borders_col_stride;
    float eaten_food = (float)borders[borders_offset];
    agents[agent_hunger_offset] += eaten_food + hunger_change_per_step;
    borders[borders_offset] = 0;
}}", lidar_angles.iter().map(f32::to_string).fold(String::new(), |a, b| a + &b + ", "),
                              lidars_len=lidar_angles.len(),
                              central_lidar_idx=central_lidar_idx,
                              steps=steps,
                              step_len=step_len,
                              hunger_change_per_step=hunger_change_per_step,
                              width=width,height=height);
        let program = Program::builder()
            .devices(context.device().clone())
            .src(src)
            .build(context.lin_alg().pro_que.context())?;
        Ok(Evol {
            lidar_count:lidar_angles.len(),
            lin_alg: context.lin_alg().clone(),
            program,
            width,
            height
        })
    }
    pub fn queue(&self) -> &Queue {
        self.lin_alg.pro_que.queue()
    }
    pub fn get_lidar_count(&self) -> usize {
        self.lidar_count
    }
    pub fn get_height(&self) -> usize {
        self.height
    }
    pub fn get_width(&self) -> usize {
        self.width
    }

    pub fn run(&self, borders:&mut Mat<u8>, agents:&mut Mat<f32>, lidars:&mut Mat<f32>) -> ocl::core::Result<()> {
        if self.width*self.height!=borders.size(){
            return Err(Error::from(format!("Evolutionary simulation has been compiled for width={} and height={} but provided world-map tensor has shape {}",self.width,self.height,borders.shape().as_shape())));
        }
        // &[agent_count, e.lidar_count, LIDAR_ATTRIBUTES]
        // &[agent_count, AGENT_ATTRIBUTES]
        // &[agent_count, ACTION_SPACE]

        if lidars.ndim() != 3 || lidars.shape()[1] != self.lidar_count || lidars.shape()[2] != LIDAR_ATTRIBUTES{
            return Err(Error::from(format!("Lidar tensor must have shape (agents, lidars, lidar_attributes) == (*, {}, {}) but got {} instead",self.lidar_count,LIDAR_ATTRIBUTES,lidars.shape().as_shape())));
        }
        if agents.ndim() != 2 || agents.shape()[1] != AGENT_ATTRIBUTES{
            return Err(Error::from(format!("Agent tensor must have shape (agents, agent_attributes) == (*, {}) but got {} instead",AGENT_ATTRIBUTES,agents.shape().as_shape())));
        }
        let agent_count = lidars.shape()[0];
        if agents.shape()[0]!=agent_count{
            return Err(Error::from(format!("Agent tensor has shape (agents, agent_attributes) == {} but lidars have mismatched shape (agents, lidars, lidar_attributes) == {}",agents.shape().as_shape(),lidars.shape().as_shape())));
        }

        KernelBuilder::new(&self.program,"evol")?
            .add_buff(borders.buffer().unwrap())?
            .add_buff(agents.buffer().unwrap())?
            .add_buff(lidars.buffer().unwrap())?
            .add_num(borders.strides()[0])?
            .add_num(borders.strides()[1])?
            .add_num(agents.strides()[0])?
            .add_num(agents.strides()[1])?
            .add_num(lidars.strides()[0])?
            .add_num(lidars.strides()[1])?
            .add_num(lidars.strides()[2])?
            .enq(self.queue(),&[agent_count])
    }
}


