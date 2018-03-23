#![feature(proc_macro)]
#![recursion_limit = "128"]

extern crate proc_macro;
#[macro_use]
extern crate quote;
extern crate syn;

extern crate accel;

use proc_macro::TokenStream;
use syn::*;
use accel::ptx_builder::*;

#[derive(Debug)]
struct Function {
    attrs: Vec<Attribute>,
    ident: Ident,
    vis: Visibility,
    block: Box<Block>,
    unsafety: Option<token::Unsafe>,
    inputs: punctuated::Punctuated<FnArg, token::Comma>,
    output: ReturnType,
    fn_token: token::Fn,
}

impl Function {
    fn parse(func: TokenStream) -> Self {
        let ItemFn {
            attrs,
            ident,
            vis,
            block,
            decl,
            unsafety,
            ..
        } = syn::parse(func.clone()).unwrap();
        let FnDecl {
            inputs,
            output,
            fn_token,
            ..
        } = { *decl };
        Function {
            attrs,
            ident,
            vis,
            block,
            unsafety,
            inputs,
            output,
            fn_token,
        }
    }

    fn input_values(&self) -> Vec<&Pat> {
        self.inputs
            .iter()
            .map(|arg| match arg {
                &FnArg::Captured(ref val) => &val.pat,
                _ => unreachable!(""),
            })
            .collect()
    }
}

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, func: TokenStream) -> TokenStream {
    let func = Function::parse(func);
    let ptx_str = func2kernel(&func);
    func2caller(&ptx_str, &func)
}

fn parse_depends(func: &Function) -> Depends {
    let mut deps = Depends::new();
    for attr in &func.attrs {
        let path = &attr.path;
        let path = &quote!{#path}.to_string();
        let tts = &attr.tts;
        let tts = &quote!{#tts}.to_string();
        let pene: &[_] = &['(', ')'];
        let dep = tts.trim_matches(pene);
        match path as &str {
            "depends" => deps.push(Crate::from_depends_str(dep)),
            "depends_path" => deps.push(Crate::from_depends_path_str(dep)),
            _ => unreachable!("Unsupported attribute: {:?}", path),
        }
    }
    deps
}

/// Convert function decorated by #[kernel] into a single `lib.rs` for PTX-builder
fn func2kernel(func: &Function) -> String {
    let vis = &func.vis;
    let fn_token = &func.fn_token;
    let ident = &func.ident;
    let unsafety = &func.unsafety;
    let inputs = &func.inputs;
    let output = &func.output;
    let block = &func.block;

    let deps = parse_depends(func);
    let crates: Vec<Ident> = deps.iter()
        .map(|c| c.name().replace("-", "_").into())
        .collect();
    let kernel = quote!{
        #![feature(abi_ptx)]
        #![no_std]
        #(extern crate #crates;), *
        #[no_mangle]
        #vis #unsafety extern "ptx-kernel" #fn_token #ident(#inputs) #output #block
    };
    compile(&kernel.to_string(), deps)
}

fn func2caller(ptx_str: &str, func: &Function) -> TokenStream {
    let vis = &func.vis;
    let fn_token = &func.fn_token;
    let ident = &func.ident;
    let inputs = &func.inputs;
    let output = &func.output;

    let input_values = func.input_values();
    let kernel_name = quote!{ #ident }.to_string();

    let caller = quote!{
        mod ptx_mod {
            use ::std::cell::RefCell;
            use ::accel::module::Module;
            thread_local! {
                #[allow(non_upper_case_globals)]
                pub static #ident: RefCell<Module> = RefCell::new(Module::from_str(#ptx_str).expect("Load module failed"));
            }
        }
        #vis #fn_token #ident(grid: ::accel::Grid, block: ::accel::Block, #inputs) #output {
            use ::accel::kernel::void_cast;
            ptx_mod::#ident.with(|m| {
                let m = m.borrow();
                let mut kernel = m.get_kernel(#kernel_name).expect("Failed to get Kernel");
                let mut args = [#(void_cast(&#input_values)),*];
                unsafe { kernel.launch(args.as_mut_ptr(), grid, block).exepct("Failed to launch kernel") };
            })
        }
    };
    caller.into()
}
