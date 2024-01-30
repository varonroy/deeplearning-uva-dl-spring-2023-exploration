use burn::{
    module::Module,
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        loss::CrossEntropyLoss,
        Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        PositionalEncoding, PositionalEncodingConfig, ReLU,
    },
    tensor::{
        backend::{AutodiffBackend, Backend},
        Int, Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};
use dl_utils::nn::ReLUConfig;
use dl_utils::{InitFns, InitFnsB};
use dl_utils_burn_sequential::SequentialForward;

use crate::data::MyBatch;

#[derive(Debug, Module, SequentialForward)]
#[dims(3, 3)]
pub struct EncoderBlockLinear<B: Backend> {
    linear_i: Linear<B>,
    dropout: Dropout,
    relu: ReLU,
    linear_o: Linear<B>,
}

#[derive(Debug, Module, SequentialForward)]
// #[dims(3, 3)]
#[manual_forward]
pub struct EncoderBlock<B: Backend> {
    attn: MultiHeadAttention<B>,
    linear: EncoderBlockLinear<B>,
    norm_1: LayerNorm<B>,
    norm_2: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> EncoderBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let attn = self.attn.forward(MhaInput::self_attn(x.clone())).context;
        let attn = self.dropout.forward(attn);
        let x = x.clone() + attn;
        let x = self.norm_1.forward(x);

        let linear_out = self.linear.forward(x.clone());
        let x = x + self.dropout.forward(linear_out);
        let x = self.norm_2.forward(x);
        x
    }
}

impl EncoderBlockConfig {
    fn new2(input_dim: usize, dim_feedforward: usize, num_heads: usize, dropout: f64) -> Self {
        Self {
            attn: MultiHeadAttentionConfig::new(input_dim, num_heads),
            linear: EncoderBlockLinearConfig::new(
                LinearConfig::new(input_dim, dim_feedforward),
                DropoutConfig::new(dropout),
                ReLUConfig::new(),
                LinearConfig::new(dim_feedforward, input_dim),
            ),
            norm_1: LayerNormConfig::new(input_dim),
            norm_2: LayerNormConfig::new(input_dim),
            dropout: DropoutConfig::new(dropout),
        }
    }
}

#[derive(Debug, Module, SequentialForward)]
#[dims(3, 3)]
pub struct Encoder<B: Backend> {
    blocks: Vec<EncoderBlock<B>>,
}

impl EncoderConfig {
    pub fn new2(
        num_layers: usize,
        input_dim: usize,
        dim_feedforward: usize,
        num_heads: usize,
        dropout: f64,
    ) -> Self {
        Self {
            blocks: (0..num_layers)
                .map(|_| EncoderBlockConfig::new2(input_dim, dim_feedforward, num_heads, dropout))
                .collect(),
        }
    }
}

#[derive(Debug, Module, SequentialForward)]
#[dims(3, 3)]
pub struct ModelInput<B: Backend> {
    dropout: Dropout,
    linear: Linear<B>,
    pos: PositionalEncoding<B>,
}

#[derive(Debug, Module, SequentialForward)]
#[dims(3, 3)]
pub struct ModelOutput<B: Backend> {
    linear_i: Linear<B>,
    norm: LayerNorm<B>,
    relu: ReLU,
    dropout: Dropout,
    linear_o: Linear<B>,
}

#[derive(Debug, Module, SequentialForward)]
#[dims(3, 3)]
pub struct Model<B: Backend> {
    input: ModelInput<B>,
    encoder: Encoder<B>,
    output: ModelOutput<B>,
}

impl<B: Backend> Model<B> {
    fn forward_classification(
        &self,
        x: Tensor<B, 3>,
        label: Tensor<B, 2, Int>,
    ) -> ClassificationOutput<B> {
        let batch_size = x.shape().dims[0];
        let seq_length = x.shape().dims[1];
        let num_classes = x.shape().dims[2];

        let out = self.forward(x);

        let out = out.reshape([batch_size * seq_length as usize, num_classes as usize]);
        let label = label.reshape([batch_size * seq_length]);

        let loss = CrossEntropyLoss::new(None).forward(out.clone(), label.clone());
        ClassificationOutput::new(loss, out, label)
    }
}

impl ModelConfig {
    pub fn new2(num_classes: usize) -> Self {
        let input_dim = num_classes;
        let model_dim = 32;
        let num_heads = 1;
        let num_classes = num_classes;
        let num_layers = 1;
        let dropout = 0.0;
        // let lr = 5e-4;
        // let warmup = 50;

        Self::new(
            ModelInputConfig::new(
                DropoutConfig::new(dropout),
                LinearConfig::new(input_dim, model_dim),
                PositionalEncodingConfig::new(model_dim),
            ),
            EncoderConfig::new2(num_layers, model_dim, 2 * model_dim, num_heads, dropout),
            ModelOutputConfig::new(
                LinearConfig::new(model_dim, model_dim),
                LayerNormConfig::new(model_dim),
                ReLUConfig::new(),
                DropoutConfig::new(dropout),
                LinearConfig::new(model_dim, num_classes),
            ),
        )
    }
}

impl<B: AutodiffBackend> TrainStep<MyBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MyBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.item, batch.label);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MyBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MyBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.item, batch.label)
    }
}

#[cfg(test)]
mod tests {
    use super::ModelConfig;
    use burn::tensor::{backend::Backend, Float, Tensor};

    fn test<B: Backend>() {
        let num_classes = 10;
        let model = ModelConfig::new2(num_classes).init::<B>();
        let input: Tensor<B, 3, Float> = Tensor::zeros([8, 32, num_classes]);
        let output = model.forward(input);
        assert_eq!(output.shape().dims, [8, 32, num_classes]);
    }

    #[test]
    fn sanity_ndarray() {
        test::<burn::backend::NdArray>();
    }

    #[test]
    fn sanity_tch_cpu() {
        test::<burn::backend::LibTorch>();
    }
}
