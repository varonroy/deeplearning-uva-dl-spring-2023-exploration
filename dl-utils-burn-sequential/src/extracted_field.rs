use quote::quote;
use syn::{Field, Ident, Type};

use crate::utils::*;

pub enum ExtractedType<'a> {
    Ty(&'a Type),
    Optional(Box<ExtractedType<'a>>),
    Vec(Box<ExtractedType<'a>>),
}

impl<'a> ExtractedType<'a> {
    fn from_ty(ty: &'a Type) -> Self {
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
                Self::Ty(ty)
            }
        } else {
            Self::Ty(ty)
        }
    }

    /// Generate a code that has in its scope the tensor `x`,
    /// and the current layer (`layer), and returns the code that
    /// forwards`x` through that layer and returns `x`.
    fn create_forward_code(&self) -> proc_macro2::TokenStream {
        match self {
            Self::Ty(_ty) => {
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
        }
    }

    fn create_config_field_type(&self) -> proc_macro2::TokenStream {
        match self {
            Self::Ty(ty) => {
                let ty_name = ty_name(ty);
                let ty_name = Ident::new(&format!("{ty_name}Config"), ty_name.span());
                quote! {#ty_name}
            }
            Self::Optional(ty) => {
                let inner = ty.create_config_field_type();
                quote! { std::option::Option<#inner> }
            }
            Self::Vec(ty) => {
                let inner = ty.create_config_field_type();
                quote! { std::vec::Vec<#inner> }
            }
        }
    }

    fn create_config_field_init_code(&self) -> proc_macro2::TokenStream {
        match self {
            Self::Ty(_ty) => {
                quote! { config.init() }
            }
            Self::Optional(ty) => {
                let inner = ty.create_config_field_init_code();
                quote! { config.as_ref().map(|config| {#inner} ) }
            }
            Self::Vec(ty) => {
                let inner = ty.create_config_field_init_code();
                quote! { config.iter().map(|config| {#inner}).collect() }
            }
        }
    }

    fn create_config_field_init_with_code(&self) -> proc_macro2::TokenStream {
        match self {
            Self::Ty(_ty) => {
                quote! { config.init_with(record) }
            }
            Self::Optional(ty) => {
                let inner = ty.create_config_field_init_with_code();
                quote! { config.as_ref().zip(record).map(|(config, record)| {#inner} ) }
            }
            Self::Vec(ty) => {
                let inner = ty.create_config_field_init_with_code();
                quote! { config.iter().zip(record).map(|(config, record)| {#inner} ).collect() }
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
