use std::f32::consts::PI;

use macroquad::{prelude::*, time};

fn window_config() -> Conf {
    Conf { 
        window_title: "u tell me a tung software this render".to_owned(),
        window_width: 800,
        window_height: 600,
        window_resizable: false,
        ..Default::default()
    }
}

fn barycentric(xa: f32, ya: f32, xb: f32, yb: f32, xc: f32, yc: f32, xp: f32, yp: f32) -> Vec3 {
    let x0 = xb - xa;
    let y0 = yb - ya;
    let x1 = xc - xa;
    let y1 = yc - ya;
    let x2 = xp - xa;
    let y2 = yp - ya;
	let idenom = 1. / (x0 * y1 - x1 * y0);
	let v = (x2 * y1 - x1 * y2) * idenom;
	let w = (x0 * y2 - x2 * y0) * idenom;
	vec3(1. - v - w, v, w)
}

fn inter<T: std::ops::Add<T, Output = T>>(br: Vec3, v0: T, v1: T, v2: T) -> T
where
    f32: std::ops::Mul<T, Output = T> {
    br.x * v0 + br.y * v1 + br.z * v2
}

trait Pipeline {
    fn vert(&self, vertex: (Vec3, Vec3, Vec2)) -> (Vec4, Vec3, Vec2);
    fn frag(&self, texel: (Vec3, Vec2)) -> Color;
}

struct Raster<P: Pipeline> {
    vertex_buffer: Box<[(Vec3, Vec3, Vec2)]>,
    index_buffer: Box<[usize]>,
    color_buffer: Image,
    depth_buffer: Box<[f32]>,
    width: u16,
    height: u16,
    pipeline: P,
}

impl<P: Pipeline> Raster<P> {
    fn new(width: u16, height: u16, pipeline: P) -> Raster<P> {
        Raster {
            vertex_buffer: vec![].into_boxed_slice(),
            index_buffer: vec![].into_boxed_slice(),
            color_buffer: Image::gen_image_color(width, height, color_u8!(0, 0, 0, 0)),
            depth_buffer: vec![f32::INFINITY; width as usize * height as usize].into_boxed_slice(),
            width,
            height,
            pipeline,
        }
    }
    fn clear(&mut self, color: Color) {
        self.color_buffer.get_image_data_mut().fill([(255. * color.r) as u8, (255. * color.g) as u8, (255. * color.b) as u8, (255. * color.a) as u8]);
    }
    fn render(&mut self) {
        let vts: Vec<((Vec4, Vec3, Vec2), (Vec4, Vec3, Vec2), (Vec4, Vec3, Vec2))> = self.index_buffer.chunks_exact(3).map(|idx| {
            (self.denormalize(self.pipeline.vert(self.vertex_buffer[idx[0]])),
             self.denormalize(self.pipeline.vert(self.vertex_buffer[idx[1]])),
             self.denormalize(self.pipeline.vert(self.vertex_buffer[idx[2]])))
        }).collect();
        vts.iter().for_each(|vt| {
            self.triangle(vt.0, vt.1, vt.2)
        })
    }
    fn pixel(&mut self, x: u16, y: u16, z: f32, color: Color) {
        let i = x as usize + y as usize * self.width as usize;
        if z + 1. >= -f32::EPSILON && z <= self.depth_buffer[i] {
            self.color_buffer.set_pixel(x as u32, y as u32, color);
        }
    }
    fn triangle(&mut self, v0: (Vec4, Vec3, Vec2), v1: (Vec4, Vec3, Vec2), v2: (Vec4, Vec3, Vec2)) {
        if (v0.0.x < 0. && v1.0.x < 0. && v2.0.x < 0.)
        || (v0.0.x > self.width as f32 && v1.0.x > self.width as f32 && v2.0.x > self.width as f32)
        || (v0.0.y < 0. && v1.0.y < 0. && v2.0.y < 0.)
        || (v0.0.y > self.height as f32 && v1.0.y > self.height as f32 && v2.0.y > self.height as f32)
        || (v0.0.z < -1. && v1.0.z < -1. && v2.0.z < -1.)
        || (v0.0.x * (v1.0.y - v2.0.y) + v0.0.y * (v2.0.x - v1.0.x) + v1.0.x * v2.0.y - v2.0.x * v1.0.y > 0.) { return; }
        let xbbmin = std::cmp::max(*[v0.0.x, v1.0.x, v2.0.x].map(|x| x as i32).iter().min().unwrap(), 0);
        let xbbmax = std::cmp::min(*[v0.0.x, v1.0.x, v2.0.x].map(|x| x as i32).iter().max().unwrap(), (self.width - 1) as i32);
        let ybbmin = std::cmp::max(*[v0.0.y, v1.0.y, v2.0.y].map(|x| x as i32).iter().min().unwrap(), 0);
        let ybbmax = std::cmp::min(*[v0.0.y, v1.0.y, v2.0.y].map(|x| x as i32).iter().max().unwrap(), (self.height - 1) as i32);
        let iw0 = 1. / v0.0.w;
        let iw1 = 1. / v1.0.w;
        let iw2 = 1. / v2.0.w;
        let zow0 = v0.0.z / v0.0.w;
        let zow1 = v1.0.z / v1.0.w;
        let zow2 = v2.0.z / v2.0.w;
        let now0 = v0.1 / v0.0.w;
        let now1 = v1.1 / v1.0.w;
        let now2 = v2.1 / v2.0.w;
        let tcow0 = v0.2 / v0.0.w;
        let tcow1 = v1.2 / v1.0.w;
        let tcow2 = v2.2 / v2.0.w;
        for i in ybbmin..=ybbmax {
            for j in xbbmin..=xbbmax {
                let br = barycentric(v0.0.x, v0.0.y, v1.0.x, v1.0.y, v2.0.x, v2.0.y, j as f32, i as f32);
                if [br.x, br.y, br.z].iter().any(|e| *e < -f32::EPSILON) { continue; }
                let pcw = 1. / inter(br, iw0, iw1, iw2);
                self.pixel(j as u16, i as u16,
                    pcw * inter(br, zow0, zow1, zow2),
                    self.pipeline.frag((pcw * inter(br, now0, now1, now2), pcw * inter(br, tcow0, tcow1, tcow2)))
                );
            }
        }
    }
    fn denormalize(&self, vertex: (Vec4, Vec3, Vec2)) -> (Vec4, Vec3, Vec2) {
        (vec4((vertex.0.x + 1.) * 0.5 * self.width as f32, (vertex.0.y + 1.) * 0.5 * self.height as f32, vertex.0.z, vertex.0.w), vertex.1, vertex.2)
    }
}

struct StdPl {
    texture: Image,
    model: Mat4,
    projection: Mat4
}

impl Pipeline for StdPl {
    fn vert(&self, vertex: (Vec3, Vec3, Vec2)) -> (Vec4, Vec3, Vec2) {
        let mut pos = self.projection * self.model * vec4(vertex.0.x, vertex.0.y, vertex.0.z, 1.);
        pos.x /= pos.w;
        pos.y /= pos.w;
        let norm = (self.model * vec4(vertex.1.x, vertex.1.y, vertex.1.z, 0.)).xyz();
        (pos, norm, vertex.2)
    }
    fn frag(&self, texel: (Vec3, Vec2)) -> Color {
        self.sample(texel.1)
    }
}

impl StdPl {
    fn sample(&self, texcoord: Vec2) -> Color {
        self.texture.get_pixel((texcoord.x * (self.texture.width - 1) as f32) as u32, (texcoord.y * (self.texture.height - 1) as f32) as u32)
    }
}

#[macroquad::main(window_config)]
async fn main() {
    let pipeline = StdPl {
        texture: Image::from_file_with_format(include_bytes!("../assets/textures/madoka.png"), Some(ImageFormat::Png)).expect(""),
        model: Mat4::IDENTITY,
        projection: Mat4::perspective_lh(PI * 0.25, 4./3., 0., 1000.),
    };
    let mut rast = Raster::new(800, 600, pipeline);
    rast.vertex_buffer = vec![
        (vec3(-1., -1., -1.), vec3(0., 0., -1.), vec2(0., 0.)),
        (vec3( 1., -1., -1.), vec3(0., 0., -1.), vec2(1., 0.)),
        (vec3( 1.,  1., -1.), vec3(0., 0., -1.), vec2(1., 1.)),
        (vec3(-1.,  1., -1.), vec3(0., 0., -1.), vec2(0., 1.)),
        (vec3(-1., -1.,  1.), vec3(0., 0., -1.), vec2(1., 0.)),
        (vec3( 1., -1.,  1.), vec3(0., 0., -1.), vec2(0., 0.)),
        (vec3( 1.,  1.,  1.), vec3(0., 0., -1.), vec2(0., 1.)),
        (vec3(-1.,  1.,  1.), vec3(0., 0., -1.), vec2(1., 1.)),
    ].into_boxed_slice();
    rast.index_buffer = vec![
        0, 3, 1,
        1, 3, 2,
        4, 5, 7,
        5, 6, 7,
        1, 2, 5,
        5, 2, 6,
        0, 4, 3,
        4, 7, 3
    ].into_boxed_slice();
    let render = Texture2D::from_image(&rast.color_buffer);
    render.set_filter(FilterMode::Nearest);
    let mut time = PI / 4.;
    let mut stop = false;
    let mut perp = true;
    loop {
        if is_key_pressed(KeyCode::Space) { stop = !stop; };
        if is_key_pressed(KeyCode::P) { perp = !perp; };
        rast.clear(color_u8!(0, 0, 0, 0));
        rast.pipeline.model = Mat4::from_translation(vec3(0., 0., 5.)) * Mat4::from_rotation_y(time);
        rast.pipeline.projection = if perp { 
            Mat4::perspective_lh(PI * 0.25, 4./3., 0., 1000.) 
        } 
        else {
            Mat4::orthographic_lh(-4., 4., -3., 3., 0., 1000.)
        };
        rast.render();
        render.update(&rast.color_buffer);
        draw_texture_ex(&render, 0., 0., color_u8!(255, 255, 255, 255), DrawTextureParams{
            dest_size: Some(vec2(800., 600.)),
            ..Default::default()
        });
        draw_text(format!("{}", get_fps()).as_str(), 0., 20., 32., color_u8!(255, 255, 255, 255));
        next_frame().await;
        if !stop {
            time += time::get_frame_time();
        }
    }
}
