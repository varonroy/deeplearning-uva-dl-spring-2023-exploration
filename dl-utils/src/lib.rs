use burn::tensor::backend::Backend;

pub mod builder;
pub mod nn;

pub trait InitFns {
    type Module;
    type Record;

    fn init(&self) -> Self::Module;

    fn init_with(&self, record: Self::Record) -> Self::Module;
}

pub trait InitFnsB<B: Backend> {
    type Module;
    type Record;

    fn init(&self) -> Self::Module;

    fn init_with(&self, record: Self::Record) -> Self::Module;
}

#[macro_export]
macro_rules! pipe {
    ($x:expr $(,$path:ident.$layer:ident)*$(,)?) => {{
        let x = $x;
        $(let x = $path.$layer.forward(x);)*
        x
    }};
}
