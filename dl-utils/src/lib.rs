pub mod builder;
pub mod nn;

#[macro_export]
macro_rules! pipe {
    ($x:expr $(,$path:ident.$layer:ident)*$(,)?) => {{
        let x = $x;
        $(let x = $path.$layer.forward(x);)*
        x
    }};
}
