use ocl::{ProQue, SpatialDims, flags, Platform, Device, Buffer, Error};

pub struct Evol {
    pro_que: ProQue,
    border:Buffer<u8>,
    lidar_count: usize,
    width:usize,
    height:usize
}
pub const ACTION_SPACE:usize = 2;
pub const AGENT_ATTRIBUTES:usize = 4;
pub const LIDAR_ATTRIBUTES:usize = 2;
pub struct Lidars{
    lidar_count: usize,
    buff:Buffer<f32>,
    agent_count:usize,
}
impl Lidars{
    pub fn new(e:&Evol, agent_count:usize)->Result<Self,Error>{
        let lidars_buff = e.pro_que.buffer_builder::<f32>()
            .flags(flags::MEM_READ_WRITE)
            .len( agent_count * e.lidar_count * LIDAR_ATTRIBUTES)
            .fill_val( 0f32)
            .build()?;
        Ok(Lidars{buff:lidars_buff,lidar_count:e.lidar_count,agent_count})
    }

    pub fn get_agent_count(&self)->usize{
        self.agent_count
    }

    pub fn len(&self)->usize{
        self.agent_count * self.lidar_count * LIDAR_ATTRIBUTES
    }

    pub fn read_lidars(&self, dst:&mut [f32])->Result<(),Error>{
        if dst.len() != self.len() {
            Err(Error::from(format!("Expected buffer length {} but got {}",self.len(), dst.len())))
        } else {
            unsafe {
                self.buff.cmd().read(dst).enq()
            }
        }
    }
}

pub struct Agents{
    buff:Buffer<f32>,
    agent_count:usize,
}
impl Agents{
    pub fn new(e:&Evol, agent_count:usize)->Result<Self,Error>{
        let buff = e.pro_que.buffer_builder::<f32>()
            .flags(flags::MEM_READ_WRITE)
            .len( agent_count * AGENT_ATTRIBUTES)
            .build()?;
        Ok(Agents{buff,agent_count})
    }

    pub fn get_agent_count(&self)->usize{
        self.agent_count
    }

    pub fn len(&self)->usize{
        self.agent_count * AGENT_ATTRIBUTES
    }

    pub fn read(&self, dst:&mut [f32])->Result<(),Error>{
        if dst.len() != self.len() {
            Err(Error::from(format!("Expected buffer length {} but got {}",self.len(), dst.len())))
        } else {
            unsafe {
                self.buff.cmd().read(dst).enq()
            }
        }
    }
}


pub struct Actions{
    buff:Buffer<f32>,
    agent_count:usize,
}
impl Actions{
    pub fn new(e:&Evol, agent_count:usize)->Result<Self,Error>{
        let buff = e.pro_que.buffer_builder::<f32>()
            .flags(flags::MEM_READ_WRITE)
            .len( agent_count * ACTION_SPACE)
            .build()?;
        Ok(Actions{buff,agent_count})
    }

    pub fn get_agent_count(&self)->usize{
        self.agent_count
    }

    pub fn len(&self)->usize{
        self.agent_count * AGENT_ATTRIBUTES
    }

    pub fn read(&self, dst:&mut [f32])->Result<(),Error>{
        if dst.len() != self.len() {
            Err(Error::from(format!("Expected buffer length {} but got {}",self.len(), dst.len())))
        } else {
            unsafe {
                self.buff.cmd().read(dst).enq()
            }
        }
    }
}

impl Evol {


    pub fn new_lidars(&self, agent_count:usize)->Result<Lidars,Error>{
        Lidars::new(self, agent_count)
    }

    pub fn new_agents(&self, agent_count:usize)->Result<Agents,Error>{
        Agents::new(self, agent_count)
    }

    pub fn new(borders:Vec<u8>,width:usize,height:usize, hunger_change_per_step:f32,lidar_angles:&[f32], steps:usize, step_len:f32, platform: Platform, device: Device) -> Result<Self, String> {
        if width*height!=borders.len(){
            return Err(format!("width and height are wrong"));
        }

        let mut src = format!("\
__kernel void evol(__global uchar * borders, __global float * agents, __global float * lidars){{
    float lidar_angles[] = {{{}}};
    uchar lidars_len = {lidars_len};
    uchar steps = {steps};
    float step_len = {step_len};
    float max_lidar_dist = steps*step_len;
    float hunger_change_per_step = {hunger_change_per_step};
    size_t agent_offset = get_global_id(0)*{agent_attributes};
    float x = agents[agent_offset];
    float y = agents[agent_offset+1];
    float rotation_action = agents[agent_offset+4];
    float angle = (agents[agent_offset+2] += rotation_action);

    float movement_action = agents[agent_offset+5];
    size_t lidar_offset = get_global_id(0)*lidars_len;
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
            size_t idx = ((size_t)lidar_y)*{width}+(size_t)lidar_x;
            uchar pixel = borders[idx];
            if(pixel==255 || new_dist >= max_lidar_dist){{
                break;
            }}
            seen_food += pixel;
            dist = new_dist;
        }}
        size_t idx = (lidar_offset+i)*{lidar_attributes};
        lidars[idx] = dist;
        lidars[idx+1] = seen_food;
    }}
    size_t idx = y*{width}+x;
    float eaten_food = (float)borders[idx];
    agents[agent_offset+3] += eaten_food + hunger_change_per_step;
    borders[idx] = 0;
}}", lidar_angles.iter().map(f32::to_string).fold(String::new(), |a, b| a + &b + ", "),
                              lidars_len=lidar_angles.len(),
                              steps=steps,
                              step_len=step_len,
                              hunger_change_per_step=hunger_change_per_step,
                              width=width,height=height,
                              agent_attributes=AGENT_ATTRIBUTES,
                              lidar_attributes=LIDAR_ATTRIBUTES);
        let pro_que = ProQue::builder()
            .platform(platform)
            .device(device)
            .src(src)
            .dims(SpatialDims::Unspecified)
            .build()?;
        let border = pro_que.buffer_builder::<u8>()
            .flags(flags::MEM_READ_WRITE)
            .len(borders.len())
            .copy_host_slice(borders.as_slice())
            .build()?;
        Ok(Evol {
            lidar_count:lidar_angles.len(),
            border,
            pro_que,
            width,
            height
        })
    }
    pub fn get_device(&self) -> Device {
        self.pro_que.device()
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
    pub fn get_borders(&self)->Result<Vec<u8>, Error>{
        let n = self.width*self.height;
        let mut v = Vec::with_capacity(n);
        unsafe {
            v.set_len(n);
            self.border.cmd()
                .queue(&self.pro_que.queue())
                .offset(0)
                .read(v.as_mut_slice())
                .enq()?;
        }
        Ok(v)
    }
    pub fn run(&self, agents:&mut [f32], lidars:&mut Lidars) -> Result<(), Error> {
        if lidars.lidar_count!=self.lidar_count{
            return Err(Error::from(format!("width and height are wrong")));
        }
        if agents.len() % AGENT_ATTRIBUTES != 0{
            return Err(Error::from(format!("Agent buffer {} not divisible by number of attributes {}",agents.len(), AGENT_ATTRIBUTES)));
        }
        let agent_count = agents.len() / AGENT_ATTRIBUTES;
        if agent_count != lidars.get_agent_count() {
            return Err(Error::from(format!("Lidar buffer can hold data of {} agents but agent buffer can hold {}",lidars.get_agent_count(), agent_count)));
        }


        let agent_buff = self.pro_que.buffer_builder::<f32>()
            .flags(flags::MEM_READ_WRITE)
            .len(agents.len())
            .copy_host_slice(agents)
            .build()?;

        let kernel = self.pro_que.kernel_builder("evol")
            .arg(&self.border)
            .arg(&agent_buff)
            .arg(&lidars.buff)
            .global_work_size(agent_count)
            .build()?;
        unsafe {
            kernel.cmd()
                .queue(&self.pro_que.queue())
                .enq()?;
        }
        unsafe {
            agent_buff.cmd()
                .queue(&self.pro_que.queue())
                .offset(0)
                .read(agents)
                .enq()?;
        }
        Ok(())
    }
}


