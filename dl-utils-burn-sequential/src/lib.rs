use itertools::Itertools;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Ident};

mod extracted_field;
mod utils;

use extracted_field::*;
use utils::*;

#[proc_macro_derive(SequentialForward, attributes(res, manual_forward, dims))]
pub fn sequential_forward(input: TokenStream) -> TokenStream {
    let DeriveInput {
        attrs,
        vis: _,
        ident,
        generics,
        data,
    } = parse_macro_input!(input as DeriveInput);

    let has_res = has_attribute(&attrs, "res");
    let manual_forward = has_attribute(&attrs, "manual_forward");
    let (in_dim, out_dim) = if manual_forward {
        (0, 0)
    } else {
        parse_dims_attr(&attrs)
    };

    let module_name = ident;
    let record_name = Ident::new(&format!("{module_name}Record"), module_name.span());
    let config_name = Ident::new(&format!("{module_name}Config"), module_name.span());

    let generic_idents = generics
        .params
        .iter()
        .map(|param| match param {
            syn::GenericParam::Lifetime(_) => todo!("generics: lifeitmes"),
            syn::GenericParam::Type(ty_param) => &ty_param.ident,
            syn::GenericParam::Const(_) => todo!("generics: const"),
        })
        .collect_vec();

    let data = match data {
        syn::Data::Struct(data) => data,
        _ => panic!("Only structs are supported"),
    };

    let extracted_fields = data
        .fields
        .iter()
        .map(|field| ExtractedField::from_field(&field))
        .collect_vec();

    let forward_fields = extracted_fields
        .iter()
        .map(|f| f.create_forward_code())
        .collect_vec();

    let config_fields = extracted_fields
        .iter()
        .map(|f| f.create_config_field())
        .collect_vec();

    let config_init_fn_body = extracted_fields
        .iter()
        .map(|f| f.create_config_field_init_line())
        .collect_vec();

    let config_init_with_fn_body = extracted_fields
        .iter()
        .map(|f| f.create_config_field_init_with_line())
        .collect_vec();

    let forward_fn_body = if has_res {
        quote! {
            let z =x.clone();
            let x = x;
            #(#forward_fields)*
            x + z
        }
    } else {
        quote! {
            let x = x;
            #(#forward_fields)*
            x
        }
    };

    let forward_fn = if manual_forward {
        quote! {}
    } else {
        quote! {
            pub fn forward(&self, x: burn::tensor::Tensor<B, #in_dim>) -> burn::tensor::Tensor<B, #out_dim> {
                #forward_fn_body
            }
        }
    };

    // eprintln!("{}", quote! {#(#config_init_fn_body,)*});

    let output = quote! {
        impl #generics #module_name <#(#generic_idents,)*> {
            #forward_fn
        }

        #[derive(burn::config::Config)]
        pub struct #config_name {
            #(#config_fields,)*
        }

        impl #config_name {
            pub fn init #generics (&self, device: &B::Device) -> #module_name<#(#generic_idents,)*> {
                #module_name {
                    #(#config_init_fn_body,)*
                }
            }

            pub fn init_with #generics (&self, record: #record_name<#(#generic_idents,)*> ) -> #module_name <#(#generic_idents,)*>  {
                #module_name {
                    #(#config_init_with_fn_body,)*
                }
            }
        }
    };

    // eprintln!("--------------");
    // eprintln!("{}", output);
    // eprintln!("--------------");

    output.into()
}
