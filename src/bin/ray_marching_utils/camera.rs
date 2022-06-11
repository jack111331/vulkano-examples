use vecmath::Vector3;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::descriptor_set::layout::DescriptorType;
use vulkano::device::Device;

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vector3<f32>,
    pub direction: Vector3<f32>,
    pub up: Vector3<f32>,
    pub fov: f32
}