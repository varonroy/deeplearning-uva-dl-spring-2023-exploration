use syn::{Attribute, Ident, Type};

pub fn ty_name(ty: &Type) -> &Ident {
    match ty {
        Type::Path(path) => {
            &path
                .path
                .segments
                .first()
                .expect("type path missing semgnet")
                .ident
        }
        _ => panic!("Expected type path"),
    }
}

pub fn extract_inner_type(path: &syn::Path) -> &Type {
    match &path.segments[0].arguments {
        syn::PathArguments::None => panic!("None path arguments inside Option"),
        syn::PathArguments::AngleBracketed(args) => match &args.args[0] {
            syn::GenericArgument::Type(ty) => ty,
            _ => panic!("Expected type inside Option"),
        },
        syn::PathArguments::Parenthesized(_) => {
            todo!("path arguments: Parenthesized")
        }
    }
}

pub fn has_attribute(attrs: &[Attribute], attr_name: &str) -> bool {
    for attr in attrs {
        match &attr.meta {
            syn::Meta::Path(path) => {
                if let Some(segment) = path.segments.first() {
                    if segment.ident == attr_name {
                        return true;
                    }
                }
            }
            _ => {}
        }
    }
    false
}

pub fn parse_dims_attr(attrs: &[Attribute]) -> (usize, usize) {
    fn parse_attr(attr: &Attribute) -> Option<(usize, usize)> {
        let list = match &attr.meta {
            syn::Meta::List(list) => list,
            _ => return None,
        };
        let path = &list.path;

        let dims_ident = path
            .segments
            .first()
            .map(|segment| segment.ident == "dims")
            .unwrap_or(false);
        if !dims_ident {
            return None;
        }

        let tokens = list.tokens.to_string();
        let (in_dim, out_dim) = tokens
            .trim()
            .split_once(",")
            .expect("dims should be separated by `,`. For example `#[dims(4, 3)]`");
        let in_dim: usize = in_dim
            .trim()
            .parse()
            .expect("dimension should be usize literal");
        let out_dim: usize = out_dim
            .trim()
            .parse()
            .expect("dimension should be usize literal");
        Some((in_dim, out_dim))
    }

    for attr in attrs {
        if let Some(dims) = parse_attr(attr) {
            return dims;
        }
    }

    panic!("Could not find dims attribute");
}
