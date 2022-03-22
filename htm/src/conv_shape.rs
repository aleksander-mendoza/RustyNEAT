use crate::{Idx, Shape3, VectorFieldOne, Shape, VectorFieldPartialOrd, range_contains, from_xyz, Shape2, w_idx, CpuSDR, AsUsize, top_small_k_indices, Weight};
use serde::{Serialize, Deserialize};
use std::ops::{Range, AddAssign, Div};
use std::fmt::Debug;

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct ConvShape {
    input_shape: [Idx; 3],
    //[height, width, channels]
    output_shape: [Idx; 3],
    //[height, width, channels]
    kernel: [Idx; 2],
    //[height, width]
    stride: [Idx; 2],
    //[height, width]
}

pub trait ConvShapeTrait {
    fn out_shape(&self) -> &[Idx; 3];
    fn in_shape(&self) -> &[Idx; 3];
    fn kernel(&self) -> &[Idx; 2];
    fn stride(&self) -> &[Idx; 2];
    fn output_shape(&self) -> [Idx; 3] {
        *self.out_shape()
    }
    fn input_shape(&self) -> [Idx; 3] {
        *self.in_shape()
    }
    fn kernel_column(&self) -> [Idx; 3] {
        self.kernel().add_channels(self.in_channels())
    }
    fn kernel_column_volume(&self) -> Idx {
        self.kernel_column().product()
    }
    fn in_grid(&self) -> &[Idx; 2] {
        self.in_shape().grid()
    }
    fn out_grid(&self) -> &[Idx; 2] {
        self.out_shape().grid()
    }
    fn out_width(&self) -> Idx {
        self.out_shape().width()
    }
    fn out_height(&self) -> Idx {
        self.out_shape().height()
    }
    fn out_channels(&self) -> Idx {
        self.out_shape().channels()
    }
    fn in_width(&self) -> Idx {
        self.in_shape().width()
    }
    fn in_height(&self) -> Idx {
        self.in_shape().height()
    }
    fn in_channels(&self) -> Idx {
        self.in_shape().channels()
    }
    fn out_area(&self) -> Idx {
        self.out_grid().area()
    }
    fn in_area(&self) -> Idx {
        self.in_grid().area()
    }
    fn out_volume(&self) -> Idx {
        self.out_shape().volume()
    }
    fn in_volume(&self) -> Idx {
        self.in_shape().volume()
    }
    fn kernel_offset(&self, output_pos: &[Idx; 3]) -> [Idx; 2] {
        output_pos.grid().conv_in_range_begin(self.stride())
    }
    fn pos_within_kernel(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> [Idx; 3] {
        debug_assert!(output_pos.all_lt(self.out_shape()));
        debug_assert!(input_pos.all_lt(self.in_shape()));
        debug_assert!(range_contains(&output_pos.grid().conv_in_range(self.stride(), self.kernel()), input_pos.grid()));
        debug_assert!(range_contains(&input_pos.grid().conv_out_range_clipped(self.stride(), self.kernel()), output_pos.grid()));
        ConvShape::sub_kernel_offset(input_pos, &self.kernel_offset(output_pos))
    }

    fn idx_within_kernel(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> Idx {
        self.kernel_column().idx(self.pos_within_kernel(input_pos, output_pos))
    }
    fn in_range(&self, output_column_pos: &[Idx; 2]) -> Range<[Idx; 2]> {
        assert!(output_column_pos.all_lt(self.out_grid()));
        output_column_pos.conv_in_range(self.stride(), self.kernel())
    }
    fn out_range(&self, input_pos: &[Idx; 2]) -> Range<[Idx; 2]> {
        input_pos.conv_out_range_clipped_both_sides(self.stride(), self.kernel(), self.out_grid())
    }
    fn sparse_broadcast<V>(&self, input: &CpuSDR, output: &CpuSDR, mut value:V,
                        mut kernel_column_finished:impl FnMut(V,Idx)->V,
                        mut target:impl FnMut(V,Idx)->V) {
        let input_pos: Vec<[Idx; 3]> = input.iter().map(|&i| self.in_shape().pos(i)).collect();
        let v = self.out_volume();
        let kernel_column = self.kernel_column();
        for &output_idx in output.as_slice() {
            let output_pos = self.out_shape().pos(output_idx);
            let kernel_offset = self.kernel_offset(&output_pos);
            let input_range = self.in_range(output_pos.grid());
            for (&input_idx, input_pos) in input.iter().zip(input_pos.iter()) {
                if range_contains(&input_range, input_pos.grid()) {
                    let w_index = ConvShape::w_index_(&input_pos, &kernel_offset, output_idx, &kernel_column, v);
                    value = target(value,w_index);
                }
            }
            value = kernel_column_finished(value,output_idx)
        }
    }
    fn sparse_unbiased_increment<D:Copy+AsUsize+Div<Output=D>+AddAssign>(&self, w_slice:&mut [D], epsilon:D, input: &CpuSDR, output: &CpuSDR){
        let w_to_increment:Vec<Idx> = Vec::with_capacity(input.len());
        self.sparse_broadcast(input,output,w_to_increment,|mut w_to_increment,_|{
            let plasticity = epsilon / D::from_usize(w_to_increment.len());
            for w_index in w_to_increment.iter().cloned() {
                w_slice[w_index.as_usize()] += plasticity;
            }
            w_to_increment.clear();
            w_to_increment
        },|mut w_to_increment,idx|{
            w_to_increment.push(idx);
            w_to_increment
        })
    }
    fn sparse_biased_increment<D:Copy+AddAssign>(&self, w_slice:&mut [D], epsilon:D, input: &CpuSDR, output: &CpuSDR){
        self.sparse_broadcast(input,output,(),|_,_|(),|(),idx|w_slice[idx.as_usize()]+=epsilon)
    }

}

impl ConvShapeTrait for ConvShape {
    fn out_shape(&self) -> &[Idx; 3] {
        &self.output_shape
    }
    fn in_shape(&self) -> &[Idx; 3] {
        &self.input_shape
    }
    fn kernel(&self) -> &[Idx; 2] {
        &self.kernel
    }
    fn stride(&self) -> &[Idx; 2] {
        &self.stride
    }
}

impl ConvShape {
    pub fn compose(&self, next: &ConvShape) -> ConvShape {
        assert_eq!(self.out_shape(), next.in_shape());
        let (kernel, stride) = self.stride().conv_compose(self.kernel(), next.stride(), next.kernel());
        Self {
            input_shape: self.input_shape,
            output_shape: next.output_shape,
            kernel,
            stride,
        }
    }

    pub fn new_identity(shape: [Idx; 3]) -> Self {
        Self {
            input_shape: shape,
            output_shape: shape,
            kernel: [1; 2],
            stride: [1; 2],
        }
    }
    pub fn new_linear(input: Idx, output: Idx) -> Self {
        Self {
            input_shape: from_xyz(1, 1, input),
            output_shape: from_xyz(1, 1, output),
            kernel: [1; 2],
            stride: [1; 2],
        }
    }
    pub fn new(output: [Idx; 2], kernel: [Idx; 2], stride: [Idx; 2], in_channels: Idx, out_channels: Idx) -> Self {
        Self::new_out(in_channels, output.add_channels(out_channels), kernel, stride)
    }
    pub fn concat<'a, T>(layers: &'a [T], f: impl Fn(&'a T) -> &'a Self) -> Self {
        assert_ne!(layers.len(), 0, "No layers provided!");
        let first_layer = f(&layers[0]);
        let mut out_shape = first_layer.output_shape();
        let in_shape = first_layer.input_shape();
        let kernel = first_layer.kernel;
        let stride = first_layer.stride;
        assert!(layers.iter().all(|a| f(a).in_shape().all_eq(&in_shape)), "All concatenated layers must have the same input shape!");
        assert!(layers.iter().all(|a| f(a).out_grid().all_eq(out_shape.grid())), "All concatenated layers must have the same output width and height!");
        assert!(layers.iter().all(|a| f(a).stride().all_eq(&stride)), "All concatenated layers must have the same stride!");
        assert!(layers.iter().all(|a| f(a).kernel().all_eq(&kernel)), "All concatenated layers must have the same kernel!");
        let concatenated_sum: Idx = layers.iter().map(|a| f(a).out_channels()).sum();
        *out_shape.channels_mut() = concatenated_sum;
        Self {
            input_shape: in_shape,
            output_shape: out_shape,
            kernel,
            stride,
        }
    }

    pub fn new_in(input_shape: [Idx; 3],
                  out_channels: Idx,
                  kernel: [Idx; 2],
                  stride: [Idx; 2]) -> Self {
        Self {
            input_shape,
            output_shape: input_shape.grid().conv_out_size(&stride, &kernel).add_channels(out_channels),
            kernel,
            stride,
        }
    }
    pub fn new_out(in_channels: Idx,
                   output_shape: [Idx; 3],
                   kernel: [Idx; 2],
                   stride: [Idx; 2]) -> Self {
        Self {
            input_shape: output_shape.grid().conv_in_size(&stride, &kernel).add_channels(in_channels),
            output_shape,
            kernel,
            stride,
        }
    }
    pub fn set_stride(&mut self, new_stride: [Idx; 2]) {
        let input = self.out_grid().conv_in_size(&new_stride, self.kernel());
        let input = input.add_channels(self.in_channels());
        self.input_shape = input;
        self.stride = new_stride;
    }
    pub fn sub_kernel_offset(input_pos: &[Idx; 3], offset: &[Idx; 2]) -> [Idx; 3] {
        from_xyz(input_pos.width() - offset.width(), input_pos.height() - offset.height(), input_pos.channels())
    }

    #[inline]
    pub fn w_index_(input_pos: &[Idx; 3], kernel_offset: &[Idx; 2], output_idx: Idx, kernel_column: &[Idx; 3], output_volume: Idx) -> Idx {
        let position_within_kernel_column = Self::sub_kernel_offset(input_pos, kernel_offset);
        w_idx(output_idx, kernel_column.idx(position_within_kernel_column), output_volume)
    }
    pub fn w_index(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> Idx {
        debug_assert!(output_pos.all_lt(self.out_shape()));
        debug_assert!(input_pos.all_lt(self.in_shape()));
        debug_assert!(range_contains(&output_pos.grid().conv_in_range(self.stride(), self.kernel()), input_pos.grid()));
        debug_assert!(range_contains(&input_pos.grid().conv_out_range_clipped(self.stride(), self.kernel()), output_pos.grid()));
        w_idx(self.out_shape().idx(*output_pos), self.idx_within_kernel(input_pos, output_pos), self.out_volume())
    }
}

pub trait HasShape {
    fn shape(&self) -> &[Idx; 3];
}

pub trait HasConvShape: HasShape {
    fn cshape(&self) -> &ConvShape;

    fn out_shape(&self) -> &[Idx; 3] {
        self.cshape().out_shape()
    }
    fn in_shape(&self) -> &[Idx; 3] {
        self.cshape().in_shape()
    }
    fn kernel(&self) -> &[Idx; 2] {
        self.cshape().kernel()
    }
    fn stride(&self) -> &[Idx; 2] {
        self.cshape().stride()
    }
    fn output_shape(&self) -> [Idx; 3] {
        self.cshape().output_shape()
    }
    fn input_shape(&self) -> [Idx; 3] {
        self.cshape().input_shape()
    }
    fn kernel_column(&self) -> [Idx; 3] {
        self.cshape().kernel_column()
    }
    fn kernel_column_volume(&self) -> Idx {
        self.cshape().kernel_column_volume()
    }
    fn in_grid(&self) -> &[Idx; 2] {
        self.cshape().in_grid()
    }
    fn out_grid(&self) -> &[Idx; 2] {
        self.cshape().out_grid()
    }
    fn out_width(&self) -> Idx {
        self.cshape().out_width()
    }
    fn out_height(&self) -> Idx {
        self.cshape().out_height()
    }
    fn out_channels(&self) -> Idx {
        self.cshape().out_channels()
    }
    fn in_width(&self) -> Idx {
        self.cshape().in_width()
    }
    fn in_height(&self) -> Idx {
        self.cshape().in_height()
    }
    fn in_channels(&self) -> Idx {
        self.cshape().in_channels()
    }
    fn out_area(&self) -> Idx {
        self.cshape().out_area()
    }
    fn in_area(&self) -> Idx {
        self.cshape().in_area()
    }
    fn out_volume(&self) -> Idx {
        self.cshape().out_volume()
    }
    fn in_volume(&self) -> Idx {
        self.cshape().in_volume()
    }
    fn kernel_offset(&self, output_pos: &[Idx; 3]) -> [Idx; 2] {
        self.cshape().kernel_offset(output_pos)
    }
    fn pos_within_kernel(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> [Idx; 3] {
        self.cshape().pos_within_kernel(input_pos, output_pos)
    }

    fn idx_within_kernel(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> Idx {
        self.cshape().idx_within_kernel(input_pos, output_pos)
    }
    fn in_range(&self, output_column_pos: &[Idx; 2]) -> Range<[Idx; 2]> {
        self.cshape().in_range(output_column_pos)
    }
    fn w_index(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> Idx {
        self.cshape().w_index(input_pos, output_pos)
    }
    fn out_range(&self, input_pos: &[Idx; 2]) -> Range<[Idx; 2]>{
        self.cshape().out_range(input_pos)
    }
    fn sparse_broadcast<V>(&self, input: &CpuSDR, output: &CpuSDR,value:V,
                        mut kernel_column_finished:impl FnMut(V,Idx)->V,
                        mut target:impl FnMut(V,Idx)->V){
        self.cshape().sparse_broadcast(input,output,value,kernel_column_finished,target)
    }
}

pub trait HasConvShapeMut: HasConvShape {
    fn cshape_mut(&mut self) -> &mut ConvShape;
    fn set_stride(&mut self, new_stride: [Idx; 2]) {
        self.cshape_mut().set_stride(new_stride)
    }
}