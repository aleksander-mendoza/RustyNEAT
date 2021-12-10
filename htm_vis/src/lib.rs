extern crate piston_window;
extern crate image as im;
extern crate vecmath;
extern crate camera_controllers;
#[macro_use]
extern crate gfx;
extern crate shader_version;

use piston_window::*;
use vecmath::*;
use gfx::{Slice, InstanceParams, UpdateError, Primitive};
use htm::{CpuHTM2, CpuBitset, EncoderTarget, CpuSDR, ArrayCast, VectorFieldMul, VectorFieldAdd, VectorFieldDiv, VectorFieldSub, VectorFieldPartialOrd, VectorFieldAbs, VectorFieldOne};
use gfx::memory::Usage;
use gfx::traits::FactoryExt;
use camera_controllers::Camera;
use gfx::state::Rasterizer;
//----------------------------------------
// Cube associated data

gfx_vertex_struct!( Vertex {
    a_pos: [f32; 3] = "a_pos",
    a_type: u32 = "a_type",
});

impl Vertex {
    fn new(pos: [f32; 3]) -> Vertex {
        Vertex {
            a_pos: pos,
            a_type: 0,
        }
    }
}

gfx_pipeline!( pipe {
    vbuf: gfx::InstanceBuffer<Vertex> = (),
    u_model_view_proj: gfx::Global<[[f32; 4]; 4]> = "u_model_view_proj",
    out_color: gfx::RenderTarget<::gfx::format::Srgba8> = "o_clr",
    out_depth: gfx::DepthTarget<::gfx::format::DepthStencil> =
        gfx::preset::depth::LESS_EQUAL_WRITE,
});

//----------------------------------------


const INPUT_CELL_MARGIN: f32 = 0.2;
const REGION_MARGIN: f32 = 2.;
const OUTPUT_CELL_MARGIN: f32 = 0.2;
const SCALE: f32 = 1.;
const HEIGHT: f32 = 2.;
const INACTIVE_CELL: u32 = 0;
const ACTIVE_CELL: u32 = 1;
const SELECTED_INACTIVE_CELL: u32 = 2;
const SELECTED_ACTIVE_CELL: u32 = 3;
const POTENTIAL_INACTIVE_CELL: u32 = 4;
const POTENTIAL_ACTIVE_CELL: u32 = 5;

pub fn visualise_cpu_htm2(htm: &CpuHTM2, input_sdr: &[u32], input_shapes: &[[u32; 3]], output_sdr: &[u32], output_shapes: &[[u32; 3]]) {
    assert_eq!(input_shapes.len(), output_shapes.len(), "Input and output shape counts do not match");
    use piston_window::*;
    use gfx::traits::*;
    use shader_version::Shaders;
    use shader_version::glsl::GLSL;
    use camera_controllers::{
        FirstPersonSettings,
        FirstPerson,
        CameraPerspective,
        model_view_projection,
    };

    let opengl = OpenGL::V3_2;

    let mut window: PistonWindow =
        WindowSettings::new("CpuHTM2 Visualiser", [640, 480])
            .exit_on_esc(true)
            .samples(4)
            .graphics_api(opengl)
            .build()
            .unwrap();
    window.set_capture_cursor(true);

    let mut factory = window.factory.clone();

    let mut vertex_data = Vec::new();


    let shape_dims: Vec<([f32; 3], [f32; 3], [f32; 3])> = input_shapes.iter().zip(output_shapes.iter()).map(|(input_shape, output_shape)| {
        let in_dim = input_shape.as_scalar::<f32>().mul_scalar(SCALE + INPUT_CELL_MARGIN).add_scalar(INPUT_CELL_MARGIN);
        let out_dim = output_shape.as_scalar::<f32>().mul_scalar(SCALE + OUTPUT_CELL_MARGIN).add_scalar(OUTPUT_CELL_MARGIN);
        let dim = in_dim.max(&out_dim);
        (dim, in_dim, out_dim)
    }).collect();
    let in_sum: u32 = input_shapes.iter().map(|d| d.product()).sum();
    assert_eq!(in_sum, htm.input_size(), "Htm input size is {} but the provided input shapes {:?} have total size {}", htm.input_size(), input_shapes, in_sum);
    let out_sum: u32 = output_shapes.iter().map(|d| d.product()).sum();
    assert_eq!(out_sum, htm.minicolumns_as_slice().len() as u32, "Htm has {} minicolumns but the provided output shapes {:?} have total size {}", htm.minicolumns_as_slice().len(), output_shapes, out_sum);
    let mut offset = 0f32;
    for (input_shape, (dim, in_dim, out_dim)) in input_shapes.iter().zip(shape_dims.iter()) {
        let half_dim = dim.div_scalar(2.);
        // htm.shape format is [depth, height, width]
        // OpenGL axes are [x, y, z]
        // OpenGL y is inverted
        // We want to visualize HTM in 3D using the format [x=width,y=-depth,z=height]
        let half_dim_no_height = [half_dim[2]/*x=width*/, 0., half_dim[1]/*z=height*/];
        for z in 0..input_shape[0] {
            for y in 0..input_shape[1] {
                for x in 0..input_shape[2] {
                    let mut pos = [x as f32, -(z as f32), y as f32].mul_scalar(SCALE + INPUT_CELL_MARGIN).add_scalar(INPUT_CELL_MARGIN).sub(&half_dim_no_height);
                    pos[0] += offset;
                    let v = Vertex::new(pos);
                    vertex_data.push(v);
                }
            }
        }
        offset += dim[2] + REGION_MARGIN;
    }
    let mut output_offset = vertex_data.len();
    let mut offset = 0f32;
    for (output_shape, (dim, in_dim, out_dim)) in output_shapes.iter().zip(shape_dims.iter()) {
        let half_dim = dim.div_scalar(2.);
        let half_dim_full_height = [half_dim[2], -HEIGHT - out_dim[0], half_dim[1]];
        for z in 0..output_shape[0] {
            for y in 0..output_shape[1] {
                for x in 0..output_shape[2] {
                    let mut pos = [x as f32, -(z as f32), y as f32].mul_scalar(SCALE + OUTPUT_CELL_MARGIN).add_scalar(OUTPUT_CELL_MARGIN).sub(&half_dim_full_height);
                    pos[0] += offset;
                    let v = Vertex::new(pos);
                    vertex_data.push(v);
                }
            }
        }
        offset += dim[2] + REGION_MARGIN;
    }

    for &i in input_sdr {
        vertex_data[i as usize].a_type = ACTIVE_CELL;
    }
    for &i in output_sdr {
        vertex_data[output_offset + i as usize].a_type = ACTIVE_CELL;
    }
    let vertex_data = vertex_data;// lock-in the data (no longer mutable)
    let vbuf = factory.create_constant_buffer::<Vertex>(vertex_data.len());
    window.encoder.update_buffer(&vbuf, &vertex_data, 0).unwrap();
    let glsl = opengl.to_glsl();
    println!("{:?}", glsl);
    let set = factory.create_shader_set(Shaders::new()
                                            .set(GLSL::V1_20, include_str!("../assets/cube_120.vert"))
                                            .set(GLSL::V1_50, include_str!("../assets/cube_150.vert"))
                                            .get(glsl).unwrap().as_bytes(),
                                        Shaders::new()
                                            .set(GLSL::V1_20, include_str!("../assets/cube_120.frag"))
                                            .set(GLSL::V1_50, include_str!("../assets/cube_150.frag"))
                                            .get(glsl).unwrap().as_bytes()).unwrap();
    let init = pipe::new();
    let pso = factory.create_pipeline_state(&set, Primitive::TriangleList, Rasterizer::new_fill().with_cull_back(), init).unwrap();

    let get_projection = |w: &PistonWindow| {
        let draw_size = w.window.draw_size();
        CameraPerspective {
            fov: 90.0,
            near_clip: 0.1,
            far_clip: 1000.0,
            aspect_ratio: (draw_size.width as f32) / (draw_size.height as f32),
        }.projection()
    };

    let model = vecmath::mat4_id();
    let mut cam = Camera::new(/*Whatever. Will be overwritten */ [0., 0., 0.]);
    let mut projection = get_projection(&window);
    let mut first_person = FirstPerson::new(
        [0.5, 0.5, 0.5],
        FirstPersonSettings::keyboard_wasd(),
    );

    let speed = 3f32;
    first_person.settings.speed_horizontal = speed;
    first_person.settings.speed_vertical = speed;
    let mut data = pipe::Data {
        vbuf,
        u_model_view_proj: [[0.0; 4]; 4],
        out_color: window.output_color.clone(),
        out_depth: window.output_stencil.clone(),
    };
    let mut slice = Slice::from_vertex_count(36);
    slice.instances = Some((vertex_data.len() as u32, 0));
    let mut working_copy_vertex_data = vertex_data.clone();
    let mut selected_output = output_offset;
    let mut new_selected_output = selected_output;
    while let Some(e) = window.next() {
        first_person.event(&e);
        if let Event::Input(Input::Button(ButtonArgs { state: ButtonState::Press, button, scancode }), _) = e {
            match button {
                Button::Mouse(MouseButton::Left) => {
                    let vec = first_person.position.sub(&cam.forward);
                    for i in output_offset..vertex_data.len() {
                        if vertex_data[i].a_pos.sub(&vec).abs().all_lt_scalar(SCALE / 2.) {
                            new_selected_output = i;
                            break;
                        }
                    }
                }
                Button::Keyboard(x) if Key::D1<=x && x<=Key::D5 => {
                    let x = (x as u32 - Key::D1 as u32)*2 + 1;
                    first_person.settings.speed_horizontal = speed*x as f32;
                    first_person.settings.speed_vertical = speed*x as f32;
                }
                _ => {}
            }
        }

        window.draw_3d(&e, |window| {
            let args = e.render_args().unwrap();
            if selected_output != new_selected_output {
                let prev_minicolumn_idx = selected_output - output_offset;
                let prev_minicolumn = &htm.minicolumns_as_slice()[prev_minicolumn_idx];
                for conn_idx in prev_minicolumn.connection_offset..(prev_minicolumn.connection_offset + prev_minicolumn.connection_len) {
                    let input_idx = htm.feedforward_connections_as_slice()[conn_idx as usize].input_id as usize;
                    working_copy_vertex_data[input_idx] = vertex_data[input_idx];
                }
                working_copy_vertex_data[selected_output] = vertex_data[selected_output];

                let minicolumn_idx = new_selected_output - output_offset;
                let minicolumn = &htm.minicolumns_as_slice()[minicolumn_idx];
                for conn_idx in minicolumn.connection_offset..(minicolumn.connection_offset + minicolumn.connection_len) {
                    let conn = &htm.feedforward_connections_as_slice()[conn_idx as usize];
                    let input_idx = conn.input_id as usize;
                    let mut cpy = &mut working_copy_vertex_data[input_idx];
                    cpy.a_type = 2 + 2 * (conn.permanence < htm.permanence_threshold) as u32 + (cpy.a_type == ACTIVE_CELL) as u32;
                }
                let cpy = &mut working_copy_vertex_data[new_selected_output];
                cpy.a_type = 2 + (cpy.a_type == ACTIVE_CELL) as u32;
                window.encoder.update_buffer(&data.vbuf, &working_copy_vertex_data, 0).unwrap();
                selected_output = new_selected_output;
            }

            window.encoder.clear(&window.output_color, [0., 0., 0., 1.0]);
            window.encoder.clear_depth(&window.output_stencil, 1.0);
            cam = first_person.camera(args.ext_dt);
            data.u_model_view_proj = model_view_projection(
                model,
                cam.orthogonal(),
                projection,
            );

            window.encoder.draw(&slice, &pso, &data);
        });

        if e.resize_args().is_some() {
            projection = get_projection(&window);
            data.out_color = window.output_color.clone();
            data.out_depth = window.output_stencil.clone();
        }
    }
}
