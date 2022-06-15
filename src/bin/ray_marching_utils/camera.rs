use vecmath::Vector3;
use std::iter::Flatten;

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vector3<f32>,
    pub direction: Vector3<f32>,
    pub up: Vector3<f32>,
    pub fov: f32,
}

use std::iter::IntoIterator;

impl Camera {
    pub fn as_vec(&self) -> Vec<f32> {
        [self.position.to_vec(), vec![0.0], self.direction.to_vec(), vec![0.0],
               self.up.to_vec(), vec![0.0], vec![self.fov],
            vec![0.0, 0.0, 0.0]].concat()
    }
}