use std::rc::Rc;

use image::GenericImageView;
use winit::window::Window;

fn barycentric(
    xa: f32,
    ya: f32,
    xb: f32,
    yb: f32,
    xc: f32,
    yc: f32,
    xp: f32,
    yp: f32,
) -> (f32, f32, f32) {
    let x0 = xb - xa;
    let y0 = yb - ya;
    let x1 = xc - xa;
    let y1 = yc - ya;
    let x2 = xp - xa;
    let y2 = yp - ya;
    let idenom = 1. / (x0 * y1 - x1 * y0);
    let u = (x2 * y1 - x1 * y2) * idenom;
    let v = (x0 * y2 - x2 * y0) * idenom;
    (1. - u - v, u, v)
}

#[derive(Debug, Clone, Copy)]
struct Vector {
    x: f32,
    y: f32,
    z: f32,
}

impl Vector {
    fn invert(self) -> Self {
        Vector {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    fn dot(self, v: Vector) -> f32 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    fn cross(self, v: Vector) -> Self {
        Vector {
            x: self.y * v.z - self.z * v.y,
            y: self.z * v.x - self.x * v.z,
            z: self.x * v.y - self.y * v.x,
        }
    }

    fn normalize(self) -> Self {
        let norm_sqr = self.x * self.x + self.y * self.y + self.z * self.z;
        if norm_sqr > 0. {
            let norm = norm_sqr.sqrt();
            Vector {
                x: self.x / norm,
                y: self.y / norm,
                z: self.z / norm,
            }
        } else {
            Vector {
                x: 0.,
                y: 0.,
                z: 0.,
            }
        }
    }
}

type Point = Vector;

const fn vector(x: f32, y: f32, z: f32) -> Vector {
    Vector { x, y, z }
}

#[derive(Debug, Clone, Copy)]
struct Tc {
    u: f32,
    v: f32,
}

const fn texcoord(u: f32, v: f32) -> Tc {
    Tc { u, v }
}

#[derive(Debug, Clone)]
struct Camera {
    position: Vector,
    direction: Vector,
    up: Vector,
    fov: f32,
}

impl Camera {
    fn frame(&self) -> Transform {
        let front = self.direction.normalize();
        let right = self.direction.cross(self.up);
        let up = right.cross(front);
        Transform {
            position: self.position.invert(),
            x: right,
            y: up,
            z: front,
        }
    }
}

#[derive(Debug, Clone)]
struct Transform {
    position: Vector,
    x: Vector,
    y: Vector,
    z: Vector,
}

impl Transform {
    fn apply_to_point(&self, p: Point) -> Point {
        Vector {
            x: p.dot(self.x) + self.position.x,
            y: p.dot(self.y) + self.position.y,
            z: p.dot(self.z) + self.position.z,
        }
    }
    fn apply_to_vector(&self, v: Vector) -> Vector {
        Vector {
            x: v.dot(self.x),
            y: v.dot(self.y),
            z: v.dot(self.z),
        }
    }
    fn apply_to_transform(&self, t: &Transform) -> Transform {
        Transform {
            position: self.apply_to_point(t.position),
            x: self.apply_to_vector(t.x),
            y: self.apply_to_vector(t.y),
            z: self.apply_to_vector(t.z),
        }
    }
}

struct Texture {
    width: u32,
    height: u32,
    color: Box<[u32]>,
}

impl Texture {
    fn sample(&self, tc: Tc) -> u32 {
        let u = ((self.width - 1) as f32 * tc.u.clamp(0., 1.)) as usize;
        let v = ((self.height - 1) as f32 * tc.v.clamp(0., 1.)) as usize;
        self.color[u + v * self.width as usize]
    }
}

struct Material {
    albedo: Texture,
}

#[derive(Clone)]
struct Mesh {
    vertices: Vec<(Point, Vector, Tc)>,
    indices: Vec<usize>,
}

impl Mesh {
    fn apply_transform(&self, transform: &Transform) -> Mesh {
        let mut nm = self.clone();
        for (p, n, _) in nm.vertices.iter_mut() {
            *p = transform.apply_to_point(*p);
            *n = transform.apply_to_vector(*n);
        }
        nm
    }
}

struct Scene {
    camera: Camera,
    transforms: Vec<(usize, usize, Transform)>,
    meshes: Vec<Mesh>,
    materials: Vec<Material>,
}

struct Render {
    width: u32,
    height: u32,
    color: Box<[u32]>,
}

impl Render {
    fn new(width: u32, height: u32) -> Self {
        Render {
            width,
            height,
            color: vec![0; (width * height) as usize].into_boxed_slice(),
        }
    }

    fn clear(&mut self, color: u32) {
        for pixel in self.color.iter_mut() {
            *pixel = color;
        }
    }

    fn render_scene(&mut self, scene: &Scene) {
        let fov_scale = 1. / (scene.camera.fov / 2.).tan();
        let aspect = self.width as f32 / self.height as f32;
        let mut depth =
            vec![std::f32::INFINITY; (self.width * self.height) as usize].into_boxed_slice();
        for transform in scene.transforms.iter() {
            let mesh = scene.meshes[transform.0]
                .apply_transform(&scene.camera.frame().apply_to_transform(&transform.2));
            let projected_points = mesh
                .vertices
                .iter()
                .map(|(p, _, _)| {
                    let projection_scale = -fov_scale / p.z;
                    (
                        (p.x * projection_scale + 1.) / 2. * (self.width as f32),
                        (p.y * projection_scale * aspect + 1.) / 2. * (self.height as f32),
                        -p.z,
                    )
                })
                .collect::<Vec<(f32, f32, f32)>>();
            for triangle in mesh.indices.chunks_exact(3) {
                let (_, n0, tc0) = mesh.vertices[triangle[0]];
                let (_, n1, tc1) = mesh.vertices[triangle[1]];
                let (_, n2, tc2) = mesh.vertices[triangle[2]];
                let (x0, y0, z0) = projected_points[triangle[0]];
                let (x1, y1, z1) = projected_points[triangle[1]];
                let (x2, y2, z2) = projected_points[triangle[2]];
                let i32_xs = [x0, x1, x2].map(|i| i as i32);
                let i32_ys = [y0, y1, y2].map(|i| i as i32);
                let min_x = std::cmp::max(*i32_xs.iter().min().unwrap(), 0);
                let min_y = std::cmp::max(*i32_ys.iter().min().unwrap(), 0);
                let max_x = std::cmp::min(*i32_xs.iter().max().unwrap(), self.width as i32 - 1);
                let max_y = std::cmp::min(*i32_ys.iter().max().unwrap(), self.height as i32 - 1);
                if min_x >= self.width as i32
                    || min_y >= self.height as i32
                    || max_x < 0
                    || max_y < 0
                {
                    continue;
                }
                for y in min_y..=max_y {
                    for x in min_x..=max_x {
                        let (t, u, v) = barycentric(x0, y0, x1, y1, x2, y2, x as f32, y as f32);
                        if [t, u, v].iter().any(|&i| i < -std::f32::EPSILON) {
                            continue;
                        }
                        let z = t * z0 + u * z1 + v * z2;
                        let i = (x + y * (self.width as i32)) as usize;
                        if z < depth[i] {
                            let tc = texcoord(
                                t * tc0.u + u * tc1.u + v * tc2.u,
                                t * tc0.v + u * tc1.v + v * tc2.v,
                            );
                            self.color[i] = scene.materials[transform.1].albedo.sample(tc);
                            self.color[i] = ((0xff as f32 * tc.u) as u32) << 16
                                | ((0xff as f32 * tc.v) as u32) << 8;
                            depth[i] = z;
                        }
                    }
                }
            }
        }
    }
}

struct App {
    window: Option<Rc<Window>>,
    context: Option<softbuffer::Context<Rc<Window>>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    scene: Option<Rc<Scene>>,
    render: Render,
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        use winit::window::WindowAttributes;
        self.window = Some(Rc::new(
            event_loop
                .create_window(WindowAttributes::default())
                .unwrap(),
        ));
        self.context =
            Some(softbuffer::Context::new(self.window.as_ref().unwrap().clone()).unwrap());
        self.surface = Some(
            softbuffer::Surface::new(
                self.context.as_ref().unwrap(),
                self.window.as_ref().unwrap().clone(),
            )
            .unwrap(),
        );
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        use winit::event::WindowEvent;
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                self.render.render_scene(self.scene.as_ref().unwrap());
                let (width, height) = {
                    let size = self.window.as_ref().unwrap().inner_size();
                    (size.width, size.height)
                };
                self.surface
                    .as_mut()
                    .unwrap()
                    .resize(
                        std::num::NonZeroU32::new(width).unwrap(),
                        std::num::NonZeroU32::new(height).unwrap(),
                    )
                    .unwrap();
                let mut buffer = self.surface.as_mut().unwrap().buffer_mut().unwrap();
                let pixel_size =
                    std::cmp::min(width / self.render.width, height / self.render.height);
                let padding_up = (height - pixel_size * self.render.height) / 2;
                let padding_left = (width - pixel_size * self.render.width) / 2;
                for i in 0..width * height {
                    let screen_x = i % width;
                    let screen_y = i / width;
                    if screen_x >= padding_left && screen_y >= padding_up {
                        let render_x = (screen_x - padding_left) / pixel_size;
                        let render_y = (screen_y - padding_up) / pixel_size;
                        if render_x < self.render.width && render_y < self.render.height {
                            buffer[i as usize] = self.render.color
                                [(render_x + render_y * self.render.width) as usize]
                        } else {
                            buffer[i as usize] = 0x0;
                        }
                    } else {
                        buffer[i as usize] = 0x0;
                    }
                }
                buffer.present().unwrap();
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}

impl App {
    fn update() {}
}

fn main() {
    let madoka = Texture {
        width: 256,
        height: 256,
        color: image::open("assets/textures/madoka.png")
            .unwrap()
            .pixels()
            .into_iter()
            .map(|(_, _, p)| (p[0] as u32) << 16 | (p[1] as u32) << 8 | (p[2] as u32))
            .collect::<Vec<u32>>()
            .into_boxed_slice(),
    };
    let scene = Rc::new(Scene {
        camera: Camera {
            position: vector(0., 0., 4.),
            direction: vector(0., 0., -1.).normalize(),
            up: vector(0., 1., 0.),
            fov: std::f32::consts::FRAC_PI_2,
        },
        transforms: vec![(
            0,
            0,
            Transform {
                position: vector(0., 0., 0.),
                x: vector(1., 0., 1.).normalize(),
                y: vector(0., 1., 0.),
                z: vector(-1., 0., 1.).normalize(),
            },
        )],
        meshes: vec![Mesh {
            vertices: vec![
                (vector(-1., -1., -1.), vector(0., 0., -1.), texcoord(0., 0.)),
                (vector(1., -1., -1.), vector(0., 0., -1.), texcoord(1., 0.)),
                (vector(1., 1., -1.), vector(0., 0., -1.), texcoord(1., 1.)),
                (vector(-1., 1., -1.), vector(0., 0., -1.), texcoord(0., 1.)),
                (vector(-1., -1., 1.), vector(0., 0., -1.), texcoord(1., 0.)),
                (vector(1., -1., 1.), vector(0., 0., -1.), texcoord(0., 0.)),
                (vector(1., 1., 1.), vector(0., 0., -1.), texcoord(0., 1.)),
                (vector(-1., 1., 1.), vector(0., 0., -1.), texcoord(1., 1.)),
            ],
            indices: vec![
                0, 3, 1, 1, 3, 2, 4, 5, 7, 5, 6, 7, 1, 2, 5, 5, 2, 6, 0, 4, 3, 4, 7, 3,
            ],
        }],
        materials: vec![Material { albedo: madoka }],
    });
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut app = App {
        window: None,
        context: None,
        surface: None,
        scene: Some(scene.clone()),
        render: Render::new(800, 600),
    };
    event_loop.run_app(&mut app).unwrap()
}
