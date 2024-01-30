#[macro_export]
macro_rules! conv_2d {
        ({$builder:expr}$(,)?) => {
            $builder
        };

        ({$builder:expr} , padding=$padding:expr $(, $($rest:tt)*)?) => {
            $crate::conv_2d!({ $builder.with_padding(burn::nn::PaddingConfig2d::Explicit($padding, $padding)) } $(, $($rest)*)?)
        };

        ({$builder:expr} , stride=$stride:expr $(, $($rest:tt)*)?) => {
            $crate::conv_2d!({ $builder.with_stride([$stride, $stride])} $(, $($rest)*)?)
        };

        ({$builder:expr} , bias=$bias:expr $(, $($rest:tt)*)?) => {
            $crate::conv_2d!({ $builder.with_bias($bias)} $(, $($rest)*)?)
        };

        ($c_in:expr, $c_out:expr, kernel_size = $kernel_size:expr $(, $($rest:tt)*)?) => {
            $crate::conv_2d!({burn::nn::conv::Conv2dConfig::new([$c_in, $c_out], [$kernel_size, $kernel_size])} $(, $($rest)*)?)
        };
    }

#[cfg(test)]
mod tests {
    use burn::nn::conv::Conv2dConfig;

    fn assert_eq_conv_2d_config(a: Conv2dConfig, b: Conv2dConfig) {
        assert_eq!(a.channels, b.channels);
        assert_eq!(a.kernel_size, b.kernel_size);
        assert_eq!(a.dilation, b.dilation);
        assert_eq!(a.stride, b.stride);
        assert_eq!(a.padding, b.padding);
        assert_eq!(a.groups, b.groups);
        assert_eq!(a.bias, b.bias);
    }

    #[test]
    fn conv_2d_macro() {
        assert_eq_conv_2d_config(
            crate::conv_2d!(3, 64, kernel_size = 3),
            Conv2dConfig::new([3, 64], [3, 3]),
        );

        assert_eq_conv_2d_config(
            crate::conv_2d!(3, 64, kernel_size = 3, padding = 1),
            Conv2dConfig::new([3, 64], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1)),
        );

        assert_eq_conv_2d_config(
            crate::conv_2d!(3, 64, kernel_size = 3, padding = 1, stride = 2),
            Conv2dConfig::new([3, 64], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .with_stride([2, 2]),
        );

        assert_eq_conv_2d_config(
            crate::conv_2d!(
                3,
                64,
                kernel_size = 3,
                padding = 1,
                stride = 2,
                bias = false
            ),
            Conv2dConfig::new([3, 64], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .with_stride([2, 2])
                .with_bias(false),
        );
    }
}
