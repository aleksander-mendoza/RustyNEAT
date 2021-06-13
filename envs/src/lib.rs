extern crate photon_rs;
mod utils;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{WebGlProgram, WebGlRenderingContext, WebGlShader, RequestMode, RequestInit, Request, WebGlBuffer};
use wasm_bindgen::prelude::*;
use photon_rs::native::{open_image, open_image_from_bytes};
use wasm_bindgen_futures::JsFuture;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
#[wasm_bindgen]
extern "C" {

    fn alert(s: &str);
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    // The `console.log` is quite polymorphic, so we can bind it with multiple
    // signatures. Note that we need to use `js_name` to ensure we always call
    // `log` in JS.
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn log_js(a: wasm_bindgen::JsValue);

    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn log_f32(a: f32);

    // Multiple arguments too!
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn log_many(a: &str, b: &str);
}

#[wasm_bindgen]
pub struct Env{
    gl:WebGlRenderingContext,
    borders:Vec<u8>,
    food:Vec<u8>,
    buffer:WebGlBuffer,
    x:usize,
}
const vertices: [f32; 8] = [
    -1.0,  1.0,
    1.0,  1.0,
    -1.0, -1.0,
    1.0, -1.0,
];

#[wasm_bindgen]
impl Env{

    pub fn new(borders_img:js_sys::Uint8Array)->Result<Env,JsValue>{
        let borders = borders_img.to_vec();
        let window = web_sys::window().unwrap();
        // let mut img = open_image_from_bytes(borders.as_slice()).expect("File should open");
        let document = window.document().unwrap();
        let canvas = document.get_element_by_id("canvas").unwrap();
        let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into::<web_sys::HtmlCanvasElement>()?;

        let gl = canvas.get_context("webgl")?.unwrap()
            .dyn_into::<WebGlRenderingContext>()?;

        let vert_shader = compile_shader(
            &gl,
            WebGlRenderingContext::VERTEX_SHADER,
            r#"
        attribute vec4 position;
        varying lowp vec4 vColor;
        void main() {
            float x = 1.0-position.x;
            float y = 1.0-position.y;
            float s = x*x+y*y;
            float f = 1.0 - 1.0/sqrt(s);
            vColor = vec4(0.0, f, 1.0, 1.0);
            gl_Position = position;
        }
    "#,
        )?;
        let frag_shader = compile_shader(
            &gl,
            WebGlRenderingContext::FRAGMENT_SHADER,
            r#"
        varying lowp vec4 vColor;
        void main() {
            gl_FragColor = vColor;
        }
    "#,
        )?;
        let program = link_program(&gl, &vert_shader, &frag_shader)?;
        gl.use_program(Some(&program));



        let buffer = gl.create_buffer().ok_or("failed to create buffer")?;
        gl.bind_buffer(WebGlRenderingContext::ARRAY_BUFFER, Some(&buffer));

        // Note that `Float32Array::view` is somewhat dangerous (hence the
        // `unsafe`!). This is creating a raw view into our module's
        // `WebAssembly.Memory` buffer, but if we allocate more pages for ourself
        // (aka do a memory allocation in Rust) it'll cause the buffer to change,
        // causing the `Float32Array` to be invalid.
        //
        // As a result, after `Float32Array::view` we have to be very careful not to
        // do any memory allocations before it's dropped.
        unsafe {
            let vert_array = js_sys::Float32Array::view(&vertices);

            gl.buffer_data_with_array_buffer_view(
                WebGlRenderingContext::ARRAY_BUFFER,
                &vert_array,
                WebGlRenderingContext::STATIC_DRAW,
            );
        }


        Ok(Self{
            gl,
            food:vec![0;borders.len()],
            borders,
            buffer,
            x:0
        })
    }
    pub fn draw(&mut self, delta_time:usize){
        self.gl.vertex_attrib_pointer_with_i32(0, 3, WebGlRenderingContext::FLOAT, false, 0, 0);
        self.gl.enable_vertex_attrib_array(0);
        self.x += delta_time;
        let x = self.x;
        let x = (x % 10000) as f32;
        let x = x / 10000. ;
        log_f32(x);
        self.gl.clear_color(x, 0.0, 0.0, 1.0);
        self.gl.clear(WebGlRenderingContext::COLOR_BUFFER_BIT);

        self.gl.draw_arrays(
            WebGlRenderingContext::TRIANGLES,
            0,
            (vertices.len() / 3) as i32,
        );
    }
}

//
// #[wasm_bindgen(start)]
// pub async fn start() -> Result<(), JsValue> {
//     Ok(())
// }

fn compile_shader(
    gl: &WebGlRenderingContext,
    shader_type: u32,
    source: &str,
) -> Result<WebGlShader, String> {
    let shader = gl
        .create_shader(shader_type)
        .ok_or_else(|| String::from("Unable to create shader object"))?;
    gl.shader_source(&shader, source);
    gl.compile_shader(&shader);

    if gl
        .get_shader_parameter(&shader, WebGlRenderingContext::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        Err(gl
            .get_shader_info_log(&shader)
            .unwrap_or_else(|| String::from("Unknown error creating shader")))
    }
}

fn link_program(
    gl: &WebGlRenderingContext,
    vert_shader: &WebGlShader,
    frag_shader: &WebGlShader,
) -> Result<WebGlProgram, String> {
    let program = gl
        .create_program()
        .ok_or_else(|| String::from("Unable to create shader object"))?;

    gl.attach_shader(&program, vert_shader);
    gl.attach_shader(&program, frag_shader);
    gl.link_program(&program);

    if gl
        .get_program_parameter(&program, WebGlRenderingContext::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(program)
    } else {
        Err(gl
            .get_program_info_log(&program)
            .unwrap_or_else(|| String::from("Unknown error creating program object")))
    }
}
