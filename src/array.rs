use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for max computation not implemented",
                ndim
            ))),
        }
    }

    /// Computes the minimum value(s) of the array along a specified axis or for the whole array.
    pub fn min_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for min computation not implemented",
                ndim
            ))),
        }
    }
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) =
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for max computation not implemented",
                ndim
            ))),
        }
    }

    /// Computes the minimum value(s) of the array along a specified axis or for the whole array.
    pub fn min_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for min computation not implemented",
                ndim
            ))),
        }
    }
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::any::type_name;
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T: NumruType, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T: NumruType, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }

    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }

    /// Returns the data type string for an array of `T`.
    pub fn dtype(&self) -> &'static str {
        type_name::<T>()
    }
}

pub trait NumruType: Debug + Copy + Zero + One + PartialOrd + 'static {}

impl NumruType for i8 {}
impl NumruType for i16 {}
impl NumruType for i32 {}
impl NumruType for i64 {}
impl NumruType for i128 {}
impl NumruType for u8 {}
impl NumruType for u16 {}
impl NumruType for u32 {}
impl NumruType for u64 {}
impl NumruType for u128 {}
impl NumruType for f32 {}
impl NumruType for f64 {}
impl NumruType for bool {}

#[cfg(test)]
mod tests {
    use std::f64::consts::{E, PI, TAU};

    use crate::{Dimension, Ix, Shape};

    #[test]
    fn array_creation_i64_1d() {
        let arr = arr![1, 2, 3, 4];
        let ix = Ix::<1>::new([4]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 4);
        assert_eq!(arr.shape().raw_dim().ndim(), 1);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_i64_2d() {
        let arr = arr![[1, 2], [3, 4], [5, 6]];
        let ix = Ix::<2>::new([3, 2]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 6);
        assert_eq!(arr.shape().raw_dim().ndim(), 2);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_i64_3d() {
        let arr = arr![[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]];
        let ix = Ix::<3>::new([2, 2, 3]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 12);
        assert_eq!(arr.shape().raw_dim().ndim(), 3);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_f64_1d() {
        let arr = arr![1.1, 2.2, 3.3, 4.4];
        let ix = Ix::<1>::new([4]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 4);
        assert_eq!(arr.shape().raw_dim().ndim(), 1);
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for max computation not implemented",
                ndim
            ))),
        }
    }

    /// Computes the minimum value(s) of the array along a specified axis or for the whole array.
    pub fn min_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for min computation not implemented",
                ndim
            ))),
        }
    }
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) =
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for max computation not implemented",
                ndim
            ))),
        }
    }

    /// Computes the minimum value(s) of the array along a specified axis or for the whole array.
    pub fn min_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for min computation not implemented",
                ndim
            ))),
        }
    }
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::any::type_name;
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T: NumruType, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T: NumruType, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }

    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }

    /// Returns the data type string for an array of `T`.
    pub fn dtype(&self) -> &'static str {
        type_name::<T>()
    }
}

pub trait NumruType: Debug + Copy + Zero + One + PartialOrd + 'static {}

impl NumruType for i8 {}
impl NumruType for i16 {}
impl NumruType for i32 {}
impl NumruType for i64 {}
impl NumruType for i128 {}
impl NumruType for u8 {}
impl NumruType for u16 {}
impl NumruType for u32 {}
impl NumruType for u64 {}
impl NumruType for u128 {}
impl NumruType for f32 {}
impl NumruType for f64 {}
impl NumruType for bool {}

#[cfg(test)]
mod tests {
    use std::f64::consts::{E, PI, TAU};

    use crate::{Dimension, Ix, Shape};

    #[test]
    fn array_creation_i64_1d() {
        let arr = arr![1, 2, 3, 4];
        let ix = Ix::<1>::new([4]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 4);
        assert_eq!(arr.shape().raw_dim().ndim(), 1);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_i64_2d() {
        let arr = arr![[1, 2], [3, 4], [5, 6]];
        let ix = Ix::<2>::new([3, 2]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 6);
        assert_eq!(arr.shape().raw_dim().ndim(), 2);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_i64_3d() {
        let arr = arr![[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]];
        let ix = Ix::<3>::new([2, 2, 3]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 12);
        assert_eq!(arr.shape().raw_dim().ndim(), 3);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_f64_1d() {
        let arr = arr![1.1, 2.2, 3.3, 4.4];
        let ix = Ix::<1>::new([4]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 4);
        assert_eq!(arr.shape().raw_dim().ndim(), 1);
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for max computation not implemented",
                ndim
            ))),
        }
    }

    /// Computes the minimum value(s) of the array along a specified axis or for the whole array.
    pub fn min_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for min computation not implemented",
                ndim
            ))),
        }
    }
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) =
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for max computation not implemented",
                ndim
            ))),
        }
    }

    /// Computes the minimum value(s) of the array along a specified axis or for the whole array.
    pub fn min_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for min computation not implemented",
                ndim
            ))),
        }
    }
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::any::type_name;
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T: NumruType, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T: NumruType, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }

    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }

    /// Returns the data type string for an array of `T`.
    pub fn dtype(&self) -> &'static str {
        type_name::<T>()
    }
}

pub trait NumruType: Debug + Copy + Zero + One + PartialOrd + 'static {}

impl NumruType for i8 {}
impl NumruType for i16 {}
impl NumruType for i32 {}
impl NumruType for i64 {}
impl NumruType for i128 {}
impl NumruType for u8 {}
impl NumruType for u16 {}
impl NumruType for u32 {}
impl NumruType for u64 {}
impl NumruType for u128 {}
impl NumruType for f32 {}
impl NumruType for f64 {}
impl NumruType for bool {}

#[cfg(test)]
mod tests {
    use std::f64::consts::{E, PI, TAU};

    use crate::{Dimension, Ix, Shape};

    #[test]
    fn array_creation_i64_1d() {
        let arr = arr![1, 2, 3, 4];
        let ix = Ix::<1>::new([4]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 4);
        assert_eq!(arr.shape().raw_dim().ndim(), 1);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_i64_2d() {
        let arr = arr![[1, 2], [3, 4], [5, 6]];
        let ix = Ix::<2>::new([3, 2]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 6);
        assert_eq!(arr.shape().raw_dim().ndim(), 2);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_i64_3d() {
        let arr = arr![[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]];
        let ix = Ix::<3>::new([2, 2, 3]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 12);
        assert_eq!(arr.shape().raw_dim().ndim(), 3);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_f64_1d() {
        let arr = arr![1.1, 2.2, 3.3, 4.4];
        let ix = Ix::<1>::new([4]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 4);
        assert_eq!(arr.shape().raw_dim().ndim(), 1);
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for max computation not implemented",
                ndim
            ))),
        }
    }

    /// Computes the minimum value(s) of the array along a specified axis or for the whole array.
    pub fn min_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for min computation not implemented",
                ndim
            ))),
        }
    }
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) =
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for max computation not implemented",
                ndim
            ))),
        }
    }

    /// Computes the minimum value(s) of the array along a specified axis or for the whole array.
    pub fn min_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for min computation not implemented",
                ndim
            ))),
        }
    }
}

use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for max computation not implemented",
                ndim
            ))),
        }
    }

    /// Computes the minimum value(s) of the array along a specified axis or for the whole array.
    pub fn min_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for min computation not implemented",
                ndim
            ))),
        }
    }
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) =
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for max computation not implemented",
                ndim
            ))),
        }
    }

    /// Computes the minimum value(s) of the array along a specified axis or for the whole array.
    pub fn min_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for min computation not implemented",
                ndim
            ))),
        }
    }
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::any::type_name;
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T: NumruType, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T: NumruType, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }

    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }

    /// Returns the data type string for an array of `T`.
    pub fn dtype(&self) -> &'static str {
        type_name::<T>()
    }
}

pub trait NumruType: Debug + Copy + Zero + One + PartialOrd + 'static {}

impl NumruType for i8 {}
impl NumruType for i16 {}
impl NumruType for i32 {}
impl NumruType for i64 {}
impl NumruType for i128 {}
impl NumruType for u8 {}
impl NumruType for u16 {}
impl NumruType for u32 {}
impl NumruType for u64 {}
impl NumruType for u128 {}
impl NumruType for f32 {}
impl NumruType for f64 {}
impl NumruType for bool {}

#[cfg(test)]
mod tests {
    use std::f64::consts::{E, PI, TAU};

    use crate::{Dimension, Ix, Shape};

    #[test]
    fn array_creation_i64_1d() {
        let arr = arr![1, 2, 3, 4];
        let ix = Ix::<1>::new([4]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 4);
        assert_eq!(arr.shape().raw_dim().ndim(), 1);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_i64_2d() {
        let arr = arr![[1, 2], [3, 4], [5, 6]];
        let ix = Ix::<2>::new([3, 2]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 6);
        assert_eq!(arr.shape().raw_dim().ndim(), 2);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_i64_3d() {
        let arr = arr![[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]];
        let ix = Ix::<3>::new([2, 2, 3]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 12);
        assert_eq!(arr.shape().raw_dim().ndim(), 3);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_f64_1d() {
        let arr = arr![1.1, 2.2, 3.3, 4.4];
        let ix = Ix::<1>::new([4]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 4);
        assert_eq!(arr.shape().raw_dim().ndim(), 1);
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for max computation not implemented",
                ndim
            ))),
        }
    }

    /// Computes the minimum value(s) of the array along a specified axis or for the whole array.
    pub fn min_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for min computation not implemented",
                ndim
            ))),
        }
    }
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) =
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for max computation not implemented",
                ndim
            ))),
        }
    }

    /// Computes the minimum value(s) of the array along a specified axis or for the whole array.
    pub fn min_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for min computation not implemented",
                ndim
            ))),
        }
    }
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::any::type_name;
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T: NumruType, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T: NumruType, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }

    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }

    /// Returns the data type string for an array of `T`.
    pub fn dtype(&self) -> &'static str {
        type_name::<T>()
    }
}

pub trait NumruType: Debug + Copy + Zero + One + PartialOrd + 'static {}

impl NumruType for i8 {}
impl NumruType for i16 {}
impl NumruType for i32 {}
impl NumruType for i64 {}
impl NumruType for i128 {}
impl NumruType for u8 {}
impl NumruType for u16 {}
impl NumruType for u32 {}
impl NumruType for u64 {}
impl NumruType for u128 {}
impl NumruType for f32 {}
impl NumruType for f64 {}
impl NumruType for bool {}

#[cfg(test)]
mod tests {
    use std::f64::consts::{E, PI, TAU};

    use crate::{Dimension, Ix, Shape};

    #[test]
    fn array_creation_i64_1d() {
        let arr = arr![1, 2, 3, 4];
        let ix = Ix::<1>::new([4]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 4);
        assert_eq!(arr.shape().raw_dim().ndim(), 1);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_i64_2d() {
        let arr = arr![[1, 2], [3, 4], [5, 6]];
        let ix = Ix::<2>::new([3, 2]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 6);
        assert_eq!(arr.shape().raw_dim().ndim(), 2);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_i64_3d() {
        let arr = arr![[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]];
        let ix = Ix::<3>::new([2, 2, 3]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 12);
        assert_eq!(arr.shape().raw_dim().ndim(), 3);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_f64_1d() {
        let arr = arr![1.1, 2.2, 3.3, 4.4];
        let ix = Ix::<1>::new([4]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 4);
        assert_eq!(arr.shape().raw_dim().ndim(), 1);
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for max computation not implemented",
                ndim
            ))),
        }
    }

    /// Computes the minimum value(s) of the array along a specified axis or for the whole array.
    pub fn min_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for min computation not implemented",
                ndim
            ))),
        }
    }
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) =
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for max computation not implemented",
                ndim
            ))),
        }
    }

    /// Computes the minimum value(s) of the array along a specified axis or for the whole array.
    pub fn min_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for min computation not implemented",
                ndim
            ))),
        }
    }
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::any::type_name;
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T: NumruType, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T: NumruType, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }

    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }

    /// Returns the data type string for an array of `T`.
    pub fn dtype(&self) -> &'static str {
        type_name::<T>()
    }
}

pub trait NumruType: Debug + Copy + Zero + One + PartialOrd + 'static {}

impl NumruType for i8 {}
impl NumruType for i16 {}
impl NumruType for i32 {}
impl NumruType for i64 {}
impl NumruType for i128 {}
impl NumruType for u8 {}
impl NumruType for u16 {}
impl NumruType for u32 {}
impl NumruType for u64 {}
impl NumruType for u128 {}
impl NumruType for f32 {}
impl NumruType for f64 {}
impl NumruType for bool {}

#[cfg(test)]
mod tests {
    use std::f64::consts::{E, PI, TAU};

    use crate::{Dimension, Ix, Shape};

    #[test]
    fn array_creation_i64_1d() {
        let arr = arr![1, 2, 3, 4];
        let ix = Ix::<1>::new([4]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 4);
        assert_eq!(arr.shape().raw_dim().ndim(), 1);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_i64_2d() {
        let arr = arr![[1, 2], [3, 4], [5, 6]];
        let ix = Ix::<2>::new([3, 2]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 6);
        assert_eq!(arr.shape().raw_dim().ndim(), 2);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_i64_3d() {
        let arr = arr![[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]];
        let ix = Ix::<3>::new([2, 2, 3]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 12);
        assert_eq!(arr.shape().raw_dim().ndim(), 3);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_f64_1d() {
        let arr = arr![1.1, 2.2, 3.3, 4.4];
        let ix = Ix::<1>::new([4]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 4);
        assert_eq!(arr.shape().raw_dim().ndim(), 1);
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for max computation not implemented",
                ndim
            ))),
        }
    }

    /// Computes the minimum value(s) of the array along a specified axis or for the whole array.
    pub fn min_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for min computation not implemented",
                ndim
            ))),
        }
    }
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) =
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for max computation not implemented",
                ndim
            ))),
        }
    }

    /// Computes the minimum value(s) of the array along a specified axis or for the whole array.
    pub fn min_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for min computation not implemented",
                ndim
            ))),
        }
    }
}

use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

pub trait NumruType: Debug + Copy + Zero + One + PartialOrd + 'static {}

impl NumruType for i8 {}
impl NumruType for i16 {}
impl NumruType for i32 {}
impl NumruType for i64 {}
impl NumruType for i128 {}
impl NumruType for u8 {}
impl NumruType for u16 {}
impl NumruType for u32 {}
impl NumruType for u64 {}
impl NumruType for u128 {}
impl NumruType for f32 {}
impl NumruType for f64 {}
impl NumruType for bool {}

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T: NumruType, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T: NumruType, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }

    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }

    /// Returns the data type string for an array of `T`.
    pub fn dtype(&self) -> &'static str {
        std::any::type_name::<T>()
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{E, PI, TAU};

    use crate::{Dimension, Ix, Shape};

    #[test]
    fn array_creation_i64_1d() {
        let arr = arr![1, 2, 3, 4];
        let ix = Ix::<1>::new([4]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 4);
        assert_eq!(arr.shape().raw_dim().ndim(), 1);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_i64_2d() {
        let arr = arr![[1, 2], [3, 4], [5, 6]];
        let ix = Ix::<2>::new([3, 2]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 6);
        assert_eq!(arr.shape().raw_dim().ndim(), 2);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_i64_3d() {
        let arr = arr![[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]];
        let ix = Ix::<3>::new([2, 2, 3]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 12);
        assert_eq!(arr.shape().raw_dim().ndim(), 3);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_f64_1d() {
        let arr = arr![1.1, 2.2, 3.3, 4.4];
        let ix = Ix::<1>::new([4]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 4);
        assert_eq!(arr.shape().raw_dim().ndim(), 1);
        assert_eq!(format!("{:
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for max computation not implemented",
                ndim
            ))),
        }
    }

    /// Computes the minimum value(s) of the array along a specified axis or for the whole array.
    pub fn min_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for min computation not implemented",
                ndim
            ))),
        }
    }
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) =
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

impl<T: Zero + One + Copy, D: Dimension> Array<T, D> {
    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }
}

impl<D: Dimension> Array<i64, D> {
    /// Returns the data type string for an array of `i64`.
    pub fn dtype(&self) -> &'static str {
        "int64"
    }
}

impl<D: Dimension> Array<f64, D> {
    /// Returns the data type string for an array of `f64`.
    pub fn dtype(&self) -> &'static str {
        "float64"
    }
}

impl<T, D: Dimension> Array<T, D>
where
    T: PartialOrd + Copy,
{
    /// Computes the maximum value(s) of the array along a specified axis or for the whole array.
    pub fn max_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for max computation not implemented",
                ndim
            ))),
        }
    }

    /// Computes the minimum value(s) of the array along a specified axis or for the whole array.
    pub fn min_compute(&self, axis: Option<usize>) -> Result<Vec<T>, ArrayError> {
        if self.data.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let raw_dim = self.shape.raw_dim();
        let ndim = raw_dim.ndim();

        if let Some(axis) = axis {
            if axis >= ndim {
                return Err(ArrayError::InvalidAxis(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }

        match ndim {
            1 => Ok(vec![*self
                .data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or(ArrayError::EmptyArray)?]),
            2 => {
                let rows = raw_dim.dims()[0];
                let cols = raw_dim.dims()[1];

                if let Some(axis) = axis {
                    if axis == 0 {
                        (0..cols)
                            .map(|col| {
                                (0..rows)
                                    .map(|row| self.data[row * cols + col])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    } else {
                        (0..rows)
                            .map(|row| {
                                self.data[row * cols..(row + 1) * cols]
                                    .iter()
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .map(|&v| v)
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>()
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            3 => {
                let depth = raw_dim.dims()[0];
                let rows = raw_dim.dims()[1];
                let cols = raw_dim.dims()[2];

                if let Some(axis) = axis {
                    match axis {
                        0 => (0..rows * cols)
                            .map(|i| {
                                (0..depth)
                                    .map(|d| self.data[d * rows * cols + i])
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .ok_or(ArrayError::EmptyArray)
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        1 => (0..depth)
                            .flat_map(|d| {
                                (0..cols).map(move |c| {
                                    (0..rows)
                                        .map(|r| self.data[d * rows * cols + r * cols + c])
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        2 => (0..depth)
                            .flat_map(|d| {
                                (0..rows).map(move |r| {
                                    let row_start = d * rows * cols + r * cols;
                                    self.data[row_start..row_start + cols]
                                        .iter()
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .map(|&v| v)
                                        .ok_or(ArrayError::EmptyArray)
                                })
                            })
                            .collect::<Result<Vec<T>, _>>(),
                        _ => unreachable!(),
                    }
                } else {
                    Ok(vec![*self
                        .data
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or(ArrayError::EmptyArray)?])
                }
            }
            _ => Err(ArrayError::UnimplementedDimension(format!(
                "Dimension {} for min computation not implemented",
                ndim
            ))),
        }
    }
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::any::type_name;
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T: NumruType, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T: NumruType, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Array { data, shape })
    }

    /// Returns a reference to the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }

    /// Replaces all elements in the array with zeros using num_traits::Zero.
    /// The shape and dimension of the array are preserved.
    pub fn zeros(&mut self) {
        let zero = T::zero(); // Use Zero::zero() instead of T::default()
        self.data.iter_mut().for_each(|x| *x = zero);
    }

    /// Replaces all elements in the array with ones using num_traits::One.
    /// The shape and dimension of the array are preserved.
    pub fn ones(&mut self) {
        let one = T::one(); // Use One::one() to get the one value
        self.data.iter_mut().for_each(|x| *x = one);
    }

    /// Returns the data type string for an array of `T`.
    pub fn dtype(&self) -> &'static str {
        type_name::<T>()
    }
}

pub trait NumruType: Debug + Copy + Zero + One + PartialOrd + 'static {}

impl NumruType for i8 {}
impl NumruType for i16 {}
impl NumruType for i32 {}
impl NumruType for i64 {}
impl NumruType for i128 {}
impl NumruType for u8 {}
impl NumruType for u16 {}
impl NumruType for u32 {}
impl NumruType for u64 {}
impl NumruType for u128 {}
impl NumruType for f32 {}
impl NumruType for f64 {}
impl NumruType for bool {}

#[cfg(test)]
mod tests {
    use std::f64::consts::{E, PI, TAU};

    use crate::{Dimension, Ix, Shape};

    #[test]
    fn array_creation_i64_1d() {
        let arr = arr![1, 2, 3, 4];
        let ix = Ix::<1>::new([4]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 4);
        assert_eq!(arr.shape().raw_dim().ndim(), 1);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_i64_2d() {
        let arr = arr![[1, 2], [3, 4], [5, 6]];
        let ix = Ix::<2>::new([3, 2]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 6);
        assert_eq!(arr.shape().raw_dim().ndim(), 2);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_i64_3d() {
        let arr = arr![[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]];
        let ix = Ix::<3>::new([2, 2, 3]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 12);
        assert_eq!(arr.shape().raw_dim().ndim(), 3);
        assert_eq!(format!("{:?}", arr.shape()), format!("{:?}", shape));
    }

    #[test]
    fn array_creation_f64_1d() {
        let arr = arr![1.1, 2.2, 3.3, 4.4];
        let ix = Ix::<1>::new([4]);
        let shape = Shape::new(ix);

        assert_eq!(arr.shape().raw_dim().size(), 4);
        assert_eq!(arr.shape().raw_dim().ndim(), 1);
use num_traits::{One, Zero};

use crate::ArrayError;
use crate::{Dimension, Shape};
use std::fmt::Debug;

/// Represents a multi-dimensional array with elements of type `T` and dimension `D`.
#[derive(Debug)]
pub struct Array<T, D: Dimension> {
    data: Vec<T>,
    shape: Shape<D>,
}

impl<T, D: Dimension> Array<T, D> {
    /// Constructs a new `Array` from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Result<Self, ArrayError> {
        let expected_size = shape.size();
        if data.len() != expected_size {
            return Err(ArrayError::DimensionMismatch {
                expected: expected_size,
