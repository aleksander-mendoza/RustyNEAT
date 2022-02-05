use crate::{Idx, Shape3, VectorFieldOne, Shape, VectorFieldPartialOrd, range_contains, from_xyz, Shape2, w_idx};
use serde::{Serialize, Deserialize};
use std::ops::Range;

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

impl ConvShape {
    pub fn compose(&self,next:&ConvShape)->ConvShape{
        assert_eq!(self.out_shape(),next.in_shape());
        let (kernel,stride) = self.stride().conv_compose(self.kernel(),next.stride(),next.kernel());
        Self{
            input_shape: self.input_shape,
            output_shape: next.output_shape,
            kernel,
            stride
        }
    }
    pub fn in_range(&self, output_column_pos:&[Idx;2])->Range<[Idx;2]>{
        assert!(output_column_pos.all_lt(self.out_grid()));
        output_column_pos.conv_in_range(self.stride(),self.kernel())
    }
    pub fn new_identity(shape: [Idx; 3])->Self{
        Self{
            input_shape: shape,
            output_shape: shape,
            kernel: [1;2],
            stride: [1;2]
        }
    }
    pub fn new(output: [Idx; 2], kernel: [Idx; 2], stride: [Idx; 2], in_channels: Idx, out_channels: Idx)->Self{
        Self::new_out(in_channels,output.add_channels(out_channels),kernel,stride)
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
    pub fn output_shape(&self)->[Idx;3]{
        self.output_shape
    }
    pub fn input_shape(&self)->[Idx;3]{
        self.input_shape
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
    pub fn out_shape(&self) -> &[Idx; 3] {
        &self.output_shape
    }
    pub fn in_shape(&self) -> &[Idx; 3] {
        &self.input_shape
    }
    pub fn kernel(&self) -> &[Idx; 2] {
        &self.kernel
    }
    pub fn stride(&self) -> &[Idx; 2] {
        &self.stride
    }
    pub fn kernel_column(&self) -> [Idx; 3] {
        self.kernel().add_channels(self.in_channels())
    }
    pub fn kernel_column_volume(&self) -> Idx {
        self.kernel_column().product()
    }
    pub fn in_grid(&self) -> &[Idx; 2] {
        self.in_shape().grid()
    }
    pub fn out_grid(&self) -> &[Idx; 2] {
        self.out_shape().grid()
    }
    pub fn out_width(&self) -> Idx {
        self.out_shape().width()
    }
    pub fn out_height(&self) -> Idx {
        self.out_shape().height()
    }
    pub fn out_channels(&self) -> Idx {
        self.out_shape().channels()
    }
    pub fn in_width(&self) -> Idx {
        self.in_shape().width()
    }
    pub fn in_height(&self) -> Idx {
        self.in_shape().height()
    }
    pub fn in_channels(&self) -> Idx {
        self.in_shape().channels()
    }
    pub fn out_area(&self) -> Idx {
        self.out_grid().product()
    }
    pub fn in_area(&self) -> Idx {
        self.in_grid().product()
    }
    pub fn out_volume(&self) -> Idx {
        self.out_shape().product()
    }
    pub fn in_volume(&self) -> Idx {
        self.in_shape().product()
    }
    pub fn kernel_offset(&self, output_pos: &[Idx; 3]) -> [Idx; 2] {
        output_pos.grid().conv_in_range_begin(self.stride())
    }
    pub fn pos_within_kernel(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> [Idx; 3] {
        debug_assert!(output_pos.all_lt(self.out_shape()));
        debug_assert!(input_pos.all_lt(self.in_shape()));
        debug_assert!(range_contains(&output_pos.grid().conv_in_range(self.stride(), self.kernel()), input_pos.grid()));
        debug_assert!(range_contains(&input_pos.grid().conv_out_range_clipped(self.stride(), self.kernel()), output_pos.grid()));
        Self::sub_kernel_offset(input_pos, &self.kernel_offset(output_pos))
    }
    pub fn sub_kernel_offset(input_pos: &[Idx; 3], offset: &[Idx; 2]) -> [Idx; 3] {
        from_xyz(input_pos.width() - offset.width(), input_pos.height() - offset.height(), input_pos.channels())
    }

    #[inline]
    pub fn w_index_(input_pos: &[Idx; 3], kernel_offset: &[Idx; 2], output_idx: Idx, kernel_column: &[Idx; 3], output_volume: Idx) -> Idx {
        let position_within_kernel_column = Self::sub_kernel_offset(input_pos, kernel_offset);
        w_idx(output_idx, kernel_column.idx(position_within_kernel_column), output_volume)
    }
    pub fn idx_within_kernel(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> Idx {
        self.kernel_column().idx(self.pos_within_kernel(input_pos, output_pos))
    }
    pub fn w_index(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> Idx {
        debug_assert!(output_pos.all_lt(self.out_shape()));
        debug_assert!(input_pos.all_lt(self.in_shape()));
        debug_assert!(range_contains(&output_pos.grid().conv_in_range(self.stride(), self.kernel()), input_pos.grid()));
        debug_assert!(range_contains(&input_pos.grid().conv_out_range_clipped(self.stride(), self.kernel()), output_pos.grid()));
        w_idx(self.out_shape().idx(*output_pos), self.idx_within_kernel(input_pos, output_pos), self.out_volume())
    }
}