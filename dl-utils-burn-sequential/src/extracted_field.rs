use quote::quote;
use syn::{Field, Ident, Type};

use crate::utils::*;

pub struct Ty<'a> {
    ty: &'a Type,
    requires_device: bool,
    has_init_with: bool,
    alternative_init: Option<proc_macro2::TokenStream>,
}

pub enum ExtractedType<'a> {
    Ty(Ty<'a>),
    Optional(Box<ExtractedType<'a>>),
    Vec(Box<ExtractedType<'a>>),
    Tuple(Vec<ExtractedType<'a>>),
}

impl<'a> ExtractedType<'a> {
    fn from_ty(ty: &'a Type) -> Self {
        match ty {
            syn::Type::Tuple(tuple) => {
                Self::Tuple(tuple.elems.iter().map(|ty| Self::from_ty(ty)).collect())
            }
            _ => {
                let path = match ty {
                    syn::Type::Path(path) => &path.path,
                    _ => todo!("other type types"),
                };
                if let Some(segment) = path.segments.first() {
                    if segment.ident == "Option" || segment.ident == "std::option::Option" {
                        let inner_ty = extract_inner_type(&path);
                        Self::Optional(Box::new(Self::from_ty(inner_ty)))
                    } else if segment.ident == "Vec" || segment.ident == "std::option::Vec" {
                        let inner_ty = extract_inner_type(&path);
                        Self::Vec(Box::new(Self::from_ty(inner_ty)))
                    } else {
                        let ident = segment.ident.to_string().to_lowercase();
                        let requires_device = match ident.as_str() {
                            "relu" | "dropout" | "flatten" | "activation" => false,
                            x if x.contains("adaptiveavgpool") => false,
                            x if x.contains("flatten") => false,
                            x if x.contains("maxpool") => false,
                            x if x.contains("avgpool") => false,
                            _ => true,
                        };
                        let has_init_with = match ident.as_str() {
                            "dropout" | "activation" => false,
                            x if x.contains("adaptiveavgpool") => false,
                            x if x.contains("flatten") => false,
                            x if x.contains("maxpool") => false,
                            x if x.contains("avgpool") => false,
                            _ => true,
                        };
                        let alternative_init = match ident.as_str() {
                            "relu" => Some(quote! {burn::nn::ReLU::new()}),
                            "identity" => Some(quote! {dl_utils::nn::Identity::new()}),
                            _ => None,
                        };
                        Self::Ty(Ty {
                            ty,
                            requires_device,
                            has_init_with,
                            alternative_init,
                        })
                    }
                } else {
                    panic!("path segments doesn't have any segments")
                }
            }
        }
    }

    /// Generate a code that has in its scope the tensor `x`,
    /// and the current layer (`layer`), and returns the code that
    /// forwards`x` through that layer and returns `x`.
    fn create_forward_code(&self) -> proc_macro2::TokenStream {
        match self {
            Self::Ty(_) => {
                quote! { layer.forward(x) }
            }
            Self::Optional(ty) => {
                let inner = ty.create_forward_code();
                quote! {
                    if let Some(layer) = &layer {
                        #inner
                    } else {
                        x
                    }
                }
            }
            Self::Vec(ty) => {
                let inner = ty.create_forward_code();
                quote! {
                    {
                        let mut x = x;
                        for layer in layer.iter() {
                            x = {
                                let x = x;
                                #inner
                            };
                        }
                        x
                    }
                }
            }
            Self::Tuple(tys) => {
                let inners = tys.iter().map(|ty| ty.create_forward_code());
                let layers = (0..tys.len())
                    .map(|i| syn::Index::from(i))
                    .map(|i| quote! {let layer = &layer.#i;});
                quote! {
                    #(
                    let x = {
                        #layers
                        #inners
                    };
                    )*
                    x
                }
            }
        }
    }

    fn create_config_field_type(&self) -> proc_macro2::TokenStream {
        match self {
            Self::Ty(Ty {
                ty,
                alternative_init,
                ..
            }) => {
                if alternative_init.is_some() {
                    quote! {()}
                } else {
                    let ty_name = ty_name(ty);
                    let ty_name = Ident::new(&format!("{ty_name}Config"), ty_name.span());
                    quote! {#ty_name}
                }
            }
            Self::Optional(ty) => {
                let inner = ty.create_config_field_type();
                quote! { std::option::Option<#inner> }
            }
            Self::Vec(ty) => {
                let inner = ty.create_config_field_type();
                quote! { std::vec::Vec<#inner> }
            }
            Self::Tuple(tys) => {
                let inner = tys.iter().map(|ty| ty.create_config_field_type());
                quote! { (#(#inner,)*) }
            }
        }
    }

    fn create_config_field_init_code(&self) -> proc_macro2::TokenStream {
        match self {
            Self::Ty(Ty {
                requires_device,
                alternative_init,
                ..
            }) => {
                if let Some(init) = alternative_init {
                    init.clone()
                } else {
                    if *requires_device {
                        quote! { config.init(device) }
                    } else {
                        quote! { config.init() }
                    }
                }
            }
            Self::Optional(ty) => {
                let inner = ty.create_config_field_init_code();
                quote! { config.as_ref().map(|config| {#inner} ) }
            }
            Self::Vec(ty) => {
                let inner = ty.create_config_field_init_code();
                quote! { config.iter().map(|config| {#inner}).collect() }
            }
            Self::Tuple(tys) => {
                let inner = tys.iter().map(|ty| ty.create_config_field_init_code());
                let i = (0..tys.len()).map(syn::Index::from).collect::<Vec<_>>();
                quote! { (#({let config = &config.#i; #inner}),*) }
            }
        }
    }

    fn create_config_field_init_with_code(&self) -> proc_macro2::TokenStream {
        match self {
            Self::Ty(Ty {
                requires_device,
                has_init_with,
                alternative_init,
                ..
            }) => {
                if let Some(init) = alternative_init {
                    init.clone()
                } else {
                    if *has_init_with {
                        quote! { config.init_with(record) }
                    } else {
                        if *requires_device {
                            panic!("doesn't have init_with and requires a device - I don't know what to do in this case....")
                        } else {
                            quote! { config.init() }
                        }
                    }
                }
            }
            Self::Optional(ty) => {
                let inner = ty.create_config_field_init_with_code();
                quote! { config.as_ref().zip(record).map(|(config, record)| {#inner} ) }
            }
            Self::Vec(ty) => {
                let inner = ty.create_config_field_init_with_code();
                quote! { config.iter().zip(record).map(|(config, record)| {#inner} ).collect() }
            }
            Self::Tuple(tys) => {
                let inner = tys.iter().map(|ty| ty.create_config_field_init_with_code());
                let i = (0..tys.len())
                    .map(|i| syn::Index::from(i))
                    .collect::<Vec<_>>();
                quote! { (#({let config = &config.#i; let record = record.#i; #inner}),*) }
            }
        }
    }
}

pub struct ExtractedField<'a> {
    pub name: &'a Ident,
    pub ty: ExtractedType<'a>,
}

impl<'a> ExtractedField<'a> {
    pub fn from_field(field: &'a Field) -> Self {
        let name = field
            .ident
            .as_ref()
            .expect("Cannot handle fields without names");

        let ty = &field.ty;
        let ty = ExtractedType::from_ty(ty);

        Self { name, ty }
    }

    pub fn create_forward_code(&self) -> proc_macro2::TokenStream {
        let name = &self.name;
        let code = self.ty.create_forward_code();
        quote! {
            let x = {
                let layer = &self.#name;
                let x = x;
                #code
            };
        }
    }

    pub fn create_config_field(&self) -> proc_macro2::TokenStream {
        let name = &self.name;
        let ty = &self.ty.create_config_field_type();
        quote! { pub #name: #ty }
    }

    pub fn create_config_field_init_line(&self) -> proc_macro2::TokenStream {
        let name = &self.name;
        let code = &self.ty.create_config_field_init_code();
        quote! {
            #name: {
                let config = &self.#name;
                #code
            }
        }
    }

    pub fn create_config_field_init_with_line(&self) -> proc_macro2::TokenStream {
        let name = &self.name;
        let code = &self.ty.create_config_field_init_with_code();
        quote! {
            #name: {
                let config = &self.#name;
                let record = record.#name;
                #code
            }
        }
    }
}
