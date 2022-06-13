// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// Welcome to the triangle example!
//
// This is the only example that is entirely detailed. All the other examples avoid code
// duplication by using helper functions.
//
// This example assumes that you are already more or less familiar with graphics programming
// and that you want to learn Vulkan. This means that for example it won't go into details about
// what a vertex or a shader is.

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents};
use vulkano::format::{Format, ClearValue};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::render_pass::{Framebuffer, Subpass, RenderPass};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::image::{ImageUsage, SwapchainImage, AttachmentImage, ImageAccess};
use vulkano::image::view::{ImageView, ImageViewAbstract};
use vulkano::instance::Instance;
use vulkano::Version;
use vulkano::device::physical::PhysicalDevice;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, PipelineBindPoint, DynamicState, ComputePipeline, Pipeline};
use vulkano::swapchain;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::swapchain::{
    AcquireError, Swapchain,
    SwapchainCreationError,
};
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};

use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use std::collections::HashMap;

use std::sync::Arc;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::buffer::cpu_access::ReadLock;
use vulkano::descriptor_set::layout::{DescriptorDesc, DescriptorSetDesc};
use vulkano::shader::ShaderStages;
use vulkano::shader::DescriptorRequirementsIncompatible::DescriptorType;

extern crate ray_marching_utils;
use ray_marching_utils::Camera;

fn main() {
    // The first step of any Vulkan program is to create an instance.
    //
    // When we create an instance, we have to pass a list of extensions that we want to enable.
    //
    // All the window-drawing functionalities are part of non-core extensions that we need
    // to enable manually. To do so, we ask the `vulkano_win` crate for the list of extensions
    // required to draw to a window.
    let required_extensions = vulkano_win::required_extensions();
    // Now creating the instance.
    let instance = Instance::new(None, Version::V1_1, &required_extensions, None).unwrap();


    // We then choose which physical device to use.
    //
    // In a real application, there are three things to take into consideration:
    //
    // - Some devices may not support some of the optional features that may be required by your
    //   application. You should filter out the devices that don't support your app.
    //
    // - Not all devices can draw to a certain surface. Once you create your window, you have to
    //   choose a device that is capable of drawing to it.
    //
    // - You probably want to leave the choice between the remaining devices to the user.
    //
    // For the sake of the example we are just going to use the first device, which should work
    // most of the time.
    // 

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    // Some little debug infos.
    println!(
        "Using device: {} (type: {:?})",
        physical.properties().device_name,
        physical.properties().device_type
    );

    // The objective of this example is to draw a triangle on a window. To do so, we first need to
    // create the window.
    //
    // This is done by creating a `WindowBuilder` from the `winit` crate, then calling the
    // `build_vk_surface` method provided by the `VkSurfaceBuild` trait from `vulkano_win`. If you
    // ever get an error about `build_vk_surface` being undefined in one of your projects, this
    // probably means that you forgot to import this trait.
    //
    // This returns a `vulkano::swapchain::Surface` object that contains both a cross-platform winit
    // window and a cross-platform Vulkan surface that represents the surface of the window.
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    // The next step is to choose which GPU queue will execute our draw commands.
    //
    // Devices can provide multiple queues to run commands in parallel (for example a draw queue
    // and a compute queue), similar to CPU threads. This is something you have to have to manage
    // manually in Vulkan.
    //
    // In a real-life application, we would probably use at least a graphics queue and a transfers
    // queue to handle data transfers in parallel. In this example we only use one queue.
    //
    // We have to choose which queues to use early on, because we will need this info very soon.
    let queue_family = physical
        .queue_families()
        .find(|&q| {
            // We take the first queue that supports drawing to our window.
            q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
        })
        .unwrap();

    // Now initializing the device. This is probably the most important object of Vulkan.
    //
    // We have to pass five parameters when creating a device:
    //
    // - Which physical device to connect to.
    //
    // - A list of optional features and extensions that our program needs to work correctly.
    //   Some parts of the Vulkan specs are optional and must be enabled manually at device
    //   creation. In this example the only thing we are going to need is the `khr_swapchain`
    //   extension that allows us to draw to a window.
    //
    // - A list of layers to enable. This is very niche, and you will usually pass `None`.
    //
    // - The list of queues that we are going to use. The exact parameter is an iterator whose
    //   items are `(Queue, f32)` where the floating-point represents the priority of the queue
    //   between 0.0 and 1.0. The priority of the queue is a hint to the implementation about how
    //   much it should prioritize queues between one another.
    //
    // The list of created queues is returned by the function alongside with the device.
    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::none()
    };
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();

    // Since we can request multiple queues, the `queues` variable is in fact an iterator. In this
    // example we use only one queue, so we just retrieve the first and only element of the
    // iterator and throw it away.
    // here we use graphic queue family
    let queue = queues.next().unwrap();

    // Before we can draw on the surface, we have to create what is called a swapchain. Creating
    // a swapchain allocates the color buffers that will contain the image that will ultimately
    // be visible on the screen. These images are returned alongside with the swapchain.
    let (mut swapchain, images) = {
        // Querying the capabilities of the surface. When we create the swapchain we can only
        // pass values that are allowed by the capabilities.
        let caps = surface.capabilities(physical).unwrap();

        // The alpha mode indicates how the alpha value of the final image will behave. For example,
        // you can choose whether the window will be opaque or transparent.
        let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();

        // Choosing the internal format that the images will have.
        let format = caps.supported_formats[0].0;

        // The dimensions of the window, only used to initially setup the swapchain.
        // NOTE:
        // On some drivers the swapchain dimensions are specified by `caps.current_extent` and the
        // swapchain size must use these dimensions.
        // These dimensions are always the same as the window dimensions
        //
        // However other drivers dont specify a value i.e. `caps.current_extent` is `None`
        // These drivers will allow anything but the only sensible value is the window dimensions.
        //
        // Because for both of these cases, the swapchain needs to be the window dimensions, we just use that.
        let dimensions: [u32; 2] = surface.window().inner_size().into();

        // Please take a look at the docs for the meaning of the parameters we didn't mention.
        Swapchain::start(device.clone(), surface.clone())
        .num_images(caps.min_image_count)
        .format(format)
        .dimensions(dimensions)
        .usage(ImageUsage::color_attachment())
        .sharing_mode(&queue)
        .composite_alpha(composite_alpha)
        .build()
        .unwrap()
    };

    // We now create a buffer that will store the shape of our triangle.
    let vertex_buffer = {
        #[derive(Default, Debug, Clone)]
        struct Vertex {
            position: [f32; 2],
        }
        vulkano::impl_vertex!(Vertex, position);

        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            [
                Vertex {
                    position: [-1.0, -1.0],
                },
                Vertex {
                    position: [-1.0, 1.0],
                },
                Vertex {
                    position: [1.0, 1.0],
                },
                Vertex {
                    position: [-1.0, -1.0],
                },
                Vertex {
                    position: [1.0, 1.0],
                },
                Vertex {
                    position: [1.0, -1.0],
                },
            ]
            .iter()
            .cloned(),
        )
        .unwrap()
    };

    let texcoord_buffer = {
        #[derive(Default, Debug, Clone)]
        struct Texcoord {
            texcoord: [f32; 2],
        }
        vulkano::impl_vertex!(Texcoord, texcoord);

        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            [
                Texcoord {
                    texcoord: [0.0, 0.0],
                },
                Texcoord {
                    texcoord: [0.0, 1.0],
                },
                Texcoord {
                    texcoord: [1.0, 1.0],
                },
                Texcoord {
                    texcoord: [0.0, 0.0],
                },
                Texcoord {
                    texcoord: [1.0, 1.0],
                },
                Texcoord {
                    texcoord: [1.0, 0.0],
                },
            ]
                .iter()
                .cloned(),
        )
            .unwrap()
    };



    // The next step is to create the shaders.
    //
    // The raw shader creation API provided by the vulkano library is unsafe, for various reasons.
    //
    // An overview of what the `vulkano_shaders::shader!` macro generates can be found in the
    // `vulkano-shaders` crate docs. You can view them at https://docs.rs/vulkano-shaders/
    //
    // TODO: explain this in details
    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
				#version 450

				layout(location = 0) in vec2 position;
				layout(location = 1) in vec2 texcoord;

				layout(location = 0) out vec2 fTexcoord;

				void main() {
					gl_Position = vec4(position, 0.0, 1.0);
					fTexcoord = texcoord;
				}
			"
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
				#version 450
				layout(location = 0) in vec2 fTexcoord;

				layout(location = 0) out vec4 f_color;
                layout(set = 0, binding = 0) uniform sampler2D texSampler;


				void main() {

					f_color = texture(texSampler, fTexcoord);
				}
			"
        }
    }

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    // At this point, OpenGL initialization would be finished. However in Vulkan it is not. OpenGL
    // implicitly does a lot of computation whenever you draw. In Vulkan, you have to do all this
    // manually.

    // The next step is to create a *render pass*, which is an object that describes where the
    // output of the graphics pipeline will go. It describes the layout of the images
    // where the colors, depth and/or stencil information will be written.
    // NOTICE: render_pass assign output attachments = =
    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            // `color` is a custom name we give to the first and only attachment.
            color: {
                // `load: Clear` means that we ask the GPU to clear the content of this
                // attachment at the start of the drawing.
                load: Clear,
                // `store: Store` means that we ask the GPU to store the output of the draw
                // in the actual image. We could also ask it to discard the result.
                store: Store,
                // `format: <ty>` indicates the type of the format of the image. This has to
                // be one of the types of the `vulkano::format` module (or alternatively one
                // of your structs that implements the `FormatDesc` trait). Here we use the
                // same format as the swapchain.
                format: swapchain.format(),
                // TODO:
                samples: 1,
            }
        },
        pass: {
            // We use the attachment named `color` as the one and only color attachment.
            color: [color],
            // No depth-stencil attachment is indicated with empty brackets.
            depth_stencil: {}
        }
    )
    .unwrap();

    // We now create a buffer that will store the shape of our triangle.
    // We use #[repr(C)] here to force rustc to not do anything funky with our data, although for this
    // particular example, it doesn't actually change the in-memory representation.
    #[repr(C)]
    #[derive(Default, Debug, Clone)]
    struct Vertex {
        position: [f32; 2],
    }
    vulkano::impl_vertex!(Vertex, position);

    #[repr(C)]
    #[derive(Default, Debug, Clone)]
    struct Texcoord {
        texcoord: [f32; 2],
    }
    vulkano::impl_vertex!(Texcoord, texcoord);

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [800.0, 600.0],
        depth_range: 0.0..1.0,
    };

    // Before we draw we have to create what is called a pipeline. This is similar to an OpenGL
    // program, but much more specific.
    let pipeline = GraphicsPipeline::start()
            // We need to indicate the layout of the vertices.
            // The type `SingleBufferDefinition` actually contains a template parameter corresponding
            // to the type of each vertex. But in this code it is automatically inferred.
            .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>().vertex::<Texcoord>())
            // A Vulkan shader can in theory contain multiple entry points, so we have to specify
            // which one. The `main` word of `main_entry_point` actually corresponds to the name of
            // the entry point.
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            // The content of the vertex buffer describes a list of triangles.
            .input_assembly_state(InputAssemblyState::new())
            // Use a resizable viewport set to draw over the entire window
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
            // See `vertex_shader`.
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            // We have to indicate which subpass of which render pass this pipeline is going to be used
            // in. The pipeline will only be usable from this particular subpass.
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            // Now that our builder is filled, we call `build()` to obtain an actual pipeline.
            .build(device.clone())
            .unwrap();
    let compute_pipeline = {
        mod cs {
            // check data.data is input as vec4??
            vulkano_shaders::shader! {
                ty: "compute",
                src: "
                    #version 450
                    layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
                    layout(binding = 1) uniform test {
                        uint a;
                    } ubo;
                    layout(binding = 2) uniform test2 {
                        uint b;
                    } ubo2;
                    layout(set = 0, binding = 0) buffer Data {
                        vec4 data[];
                    } data;
                    void main() {
                        uint yidx = gl_GlobalInvocationID.x;
                        uint xidx = gl_GlobalInvocationID.y;
                        data.data[yidx * 800 + xidx] = vec4(float(yidx) / 600, float(xidx) / 800, 0.0, 1.0) + 0.001 * vec4(ubo.a, ubo.a, ubo.a, 0.0);
                    }
                "
            }
        }
        let shader = cs::load(device.clone()).unwrap();
        // it will automatically parse shader's buffer object descriptor
        // ubo Some(DescriptorDesc { ty: UniformBuffer, descriptor_count: 1, variable_count: false, stages: ShaderStages { vertex: false, tessellation_control: false, tessellation_evaluation: false, geometry: false, fragment: false, compute: true, raygen: false, any_hit: false, closest_hit: false, miss: false, intersection: false, callable: false }, immutable_samplers: [] }),

        ComputePipeline::new(
            device.clone(),
            shader.entry_point("main").unwrap(),
            &(),
            None,
            |dd| {
                println!("DescriptorSetDesc for compute shader: {:?}", dd);
            },
        )
        .unwrap()

    };

    let data_buffer = {
        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage {
                storage_buffer: true,
                transfer_source: true,
                ..BufferUsage::none()
            },
            false,
            (0..800*600).map(|_| (1.0_f32,0.0_f32,0.0_f32,0.0_f32)),
        ).unwrap()
    };

    let uniform_data_buffer = {
        let data = Camera {
            position: [0.0, 0.0, 0.0],
            direction: [0.0, 0.0, 0.0],
            up: [0.0, 0.0, 0.0],
            fov: 45.0
        };

        CpuAccessibleBuffer::from_data(
            device.clone(),
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::none()
            },
            false,
            data,
        ).unwrap()
    };

    let atch_usage = ImageUsage {
        // transient_attachment: true,
        input_attachment: true,
        transfer_destination: true,
        sampled: true,
        ..ImageUsage::none()
    };


    let compute_layout = compute_pipeline.layout().descriptor_set_layouts().get(0).unwrap();
    let compute_set = PersistentDescriptorSet::new(
        compute_layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone()),
            WriteDescriptorSet::buffer(1, uniform_data_buffer.clone())],
    ).unwrap();

    // TODO https://github.com/vulkano-rs/vulkano/blob/ac3e4311680ec3ed9f9d6c11db6bef8a37003a4f/examples/src/bin/deferred/frame/system.rs
    let output_image_view = ImageView::new(AttachmentImage::with_usage(
        device.clone(),
        [800, 600],
        Format::R32G32B32A32_SFLOAT,
        atch_usage,
    ).unwrap()
    ).unwrap();

    let sampler = Sampler::simple_repeat_linear(device.clone())
        .unwrap();

    let graphic_layout = pipeline.layout().descriptor_set_layouts()
        .get(0)
        .unwrap();
    // FIXME
    let graphic_set = PersistentDescriptorSet::new(
        graphic_layout.clone(),
        [WriteDescriptorSet::image_view_sampler(0, output_image_view.clone(), sampler.clone())],
    ).unwrap();


    // Dynamic viewports allow us to recreate just the viewport when the window is resized
    // Otherwise we would have to recreate the whole pipeline.
    // let mut dynamic_state: HashMap<DynamicState, bool> = HashMap::default();
    // DynamicState {
    //     LineWidth(None),
    //     viewports: None,
    //     scissors: None,
    //     compare_mask: None,
    //     write_mask: None,
    //     reference: None,
    // };

    // The render pass we created above only describes the layout of our framebuffers. Before we
    // can draw we also need to create the actual framebuffers.
    //
    // Since we need to draw to multiple images, we are going to create a different framebuffer for
    // each image.

    let mut framebuffers =
        window_size_dependent_setup(&images, render_pass.clone());

    // Initialization is finally finished!

    // In some situations, the swapchain will become invalid by itself. This includes for example
    // when the window is resized (as the images of the swapchain will no longer match the
    // window's) or, on Android, when the application went to the background and goes back to the
    // foreground.
    //
    // In this situation, acquiring a swapchain image or presenting it will return an error.
    // Rendering to an image of that swapchain will not produce any error, but may or may not work.
    // To continue rendering, we need to recreate the swapchain by creating a new swapchain.
    // Here, we remember that we need to do this for the next loop iteration.
    let mut recreate_swapchain = false;

    // In the loop below we are going to submit commands to the GPU. Submitting a command produces
    // an object that implements the `GpuFuture` trait, which holds the resources for as long as
    // they are in use by the GPU.
    //
    // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
    // that, we store the submission of the previous frame here.
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::RedrawEventsCleared => {
                // It is important to call this function from time to time, otherwise resources will keep
                // accumulating and you will eventually reach an out of memory error.
                // Calling this function polls various fences in order to determine what the GPU has
                // already processed, and frees the resources that are no longer needed.
                previous_frame_end.as_mut().unwrap().cleanup_finished();
                // Whenever the window resizes we need to recreate everything dependent on the window size.
                // In this example that includes the swapchain, the framebuffers and the dynamic state viewport.
                if recreate_swapchain {
                    // Get the new dimensions of the window.
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) =
                        match swapchain.recreate().dimensions(dimensions).build() {
                            Ok(r) => r,
                            // This error tends to happen when the user is manually resizing the window.
                            // Simply restarting the loop is the easiest way to fix this issue.
                            Err(SwapchainCreationError::UnsupportedDimensions) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    swapchain = new_swapchain;
                    // Because framebuffers contains an Arc on the old swapchain, we need to
                    // recreate framebuffers as well.
                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                    );
                    recreate_swapchain = false;
                }

                // Before we can draw on the output, we have to *acquire* an image from the swapchain. If
                // no image is available (which happens if you submit draw commands too quickly), then the
                // function will block.
                // This operation returns the index of the image that we are allowed to draw upon.
                //
                // This function can block if no image is available. The parameter is an optional timeout
                // after which the function call will return an error.
                let (image_num, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                // acquire_next_image can be successful, but suboptimal. This means that the swapchain image
                // will still work, but it may not display correctly. With some drivers this can be when
                // the window resizes, but it may not cause the swapchain to become out of date.
                if suboptimal {
                    recreate_swapchain = true;
                }

                // Specify the color to clear the framebuffer with i.e. blue
                let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];

                // In order to draw, we have to build a *command buffer*. The command buffer object holds
                // the list of commands that are going to be executed.
                //
                // Building a command buffer is an expensive operation (usually a few hundred
                // microseconds), but it is known to be a hot path in the driver and is expected to be
                // optimized.
                //
                // Note that we have to pass a queue family when we create the command buffer. The command
                // buffer will only be executable on that given queue family.
                let mut builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();
                builder
                    .bind_pipeline_compute(compute_pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        compute_pipeline.layout().clone(),
                        0,
                        compute_set.clone(),
                    )
                    .dispatch([600, 800, 1])
                    .unwrap()

                    // TODO directly copy buffer to image isn't feasible, because framebuffer's ImageView don't have transfer bit allowed
                    .copy_buffer_to_image(data_buffer.clone(), output_image_view.image().clone())
                    .unwrap()

                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .bind_vertex_buffers(1, texcoord_buffer.clone())
                    // Before we can draw, we have to *enter a render pass*. There are two methods to do
                    // this: `draw_inline` and `draw_secondary`. The latter is a bit more advanced and is
                    // not covered here.
                    //
                    // The third parameter builds the list of values to clear the attachments with. The API
                    // is similar to the list of attachments when building the framebuffers, except that
                    // only the attachments that use `load: Clear` appear in the list.
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        clear_values,
                    )
                    .unwrap()
                    .bind_pipeline_graphics(pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.layout().clone(),
                        0,
                        graphic_set.clone(),
                    )
                    // We are now inside the first subpass of the render pass. We add a draw command.
                    //
                    // The last two parameters contain the list of resources to pass to the shaders.
                    // Since we used an `EmptyPipeline` object, the objects have to be `()`.
                    .draw(
                        6,
                        1,
                        0,
                        0,
                    )
                    .unwrap()
                    // We leave the render pass by calling `draw_end`. Note that if we had multiple
                    // subpasses we could have called `next_inline` (or `next_secondary`) to jump to the
                    // next subpass.
                    .end_render_pass()
                    .unwrap();
                // Finish building the command buffer by calling `build`.

                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    // The color output is now expected to contain our triangle. But in order to show it on
                    // the screen, we have to *present* the image by calling `present`.
                    //
                    // This function does not actually present the image immediately. Instead it submits a
                    // present command at the end of the queue. This means that it will only be presented once
                    // the GPU has finished executing the command buffer that draws the triangle.
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        future.wait(None).unwrap();
                        println!("Check");
                        // let data_buffer_content = data_buffer.read().unwrap();
                        // for n in 0..65536u32 {
                        //     assert_eq!(data_buffer_content[n as usize], n * 12);
                        //     println!("{}, {}", n, data_buffer_content[n as usize]);
                        // }
                        // *control_flow = ControlFlow::Exit;
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => (),
        }
    });
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {

    images
        .iter()
        .map(|image| {
            let view = ImageView::new(image.clone()).unwrap();
            Framebuffer::start(render_pass.clone())
                .add(view)
                .unwrap()
                .build()
                .unwrap()
        })
        .collect::<Vec<_>>()
}
