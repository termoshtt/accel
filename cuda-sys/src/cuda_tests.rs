#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]
#![cfg(test)]

use super::cuda::*;

#[test]
fn bindgen_test_layout_CUuuid_st() {
    assert_eq!(::std::mem::size_of::<CUuuid_st>(), 16usize);
    assert_eq!(::std::mem::align_of::<CUuuid_st>(), 1usize);
}
#[test]
fn bindgen_test_layout_CUipcEventHandle_st() {
    assert_eq!(::std::mem::size_of::<CUipcEventHandle_st>(), 64usize);
    assert_eq!(::std::mem::align_of::<CUipcEventHandle_st>(), 1usize);
}
#[test]
fn bindgen_test_layout_CUipcMemHandle_st() {
    assert_eq!(::std::mem::size_of::<CUipcMemHandle_st>(), 64usize);
    assert_eq!(::std::mem::align_of::<CUipcMemHandle_st>(), 1usize);
}
#[test]
fn bindgen_test_layout_CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st__bindgen_ty_1() {
    assert_eq!(
        ::std::mem::size_of::<CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st__bindgen_ty_1>(),
        8usize
    );
    assert_eq!(
        ::std::mem::align_of::<CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st__bindgen_ty_1>(),
        8usize
    );
}
#[test]
fn bindgen_test_layout_CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st() {
    assert_eq!(
        ::std::mem::size_of::<CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st>(),
        40usize
    );
    assert_eq!(
        ::std::mem::align_of::<CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st>(),
        8usize
    );
}
#[test]
fn bindgen_test_layout_CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st__bindgen_ty_1() {
    assert_eq!(
        ::std::mem::size_of::<CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st__bindgen_ty_1>(),
        8usize
    );
    assert_eq!(
        ::std::mem::align_of::<CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st__bindgen_ty_1>(),
        8usize
    );
}
#[test]
fn bindgen_test_layout_CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st() {
    assert_eq!(
        ::std::mem::size_of::<CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st>(),
        40usize
    );
    assert_eq!(
        ::std::mem::align_of::<CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st>(),
        8usize
    );
}
#[test]
fn bindgen_test_layout_CUstreamBatchMemOpParams_union_CUstreamMemOpFlushRemoteWritesParams_st() {
    assert_eq!(
        ::std::mem::size_of::<CUstreamBatchMemOpParams_union_CUstreamMemOpFlushRemoteWritesParams_st>(),
        8usize
    );
    assert_eq!(
        ::std::mem::align_of::<CUstreamBatchMemOpParams_union_CUstreamMemOpFlushRemoteWritesParams_st>(),
        4usize
    );
}
#[test]
fn bindgen_test_layout_CUstreamBatchMemOpParams_union() {
    assert_eq!(
        ::std::mem::size_of::<CUstreamBatchMemOpParams_union>(),
        48usize
    );
    assert_eq!(
        ::std::mem::align_of::<CUstreamBatchMemOpParams_union>(),
        8usize
    );
}
#[test]
fn bindgen_test_layout_CUdevprop_st() {
    assert_eq!(::std::mem::size_of::<CUdevprop_st>(), 56usize);
    assert_eq!(::std::mem::align_of::<CUdevprop_st>(), 4usize);
}
#[test]
fn bindgen_test_layout_CUDA_MEMCPY2D_st() {
    assert_eq!(::std::mem::size_of::<CUDA_MEMCPY2D_st>(), 128usize);
    assert_eq!(::std::mem::align_of::<CUDA_MEMCPY2D_st>(), 8usize);
}
#[test]
fn bindgen_test_layout_CUDA_MEMCPY3D_st() {
    assert_eq!(::std::mem::size_of::<CUDA_MEMCPY3D_st>(), 200usize);
    assert_eq!(::std::mem::align_of::<CUDA_MEMCPY3D_st>(), 8usize);
}
#[test]
fn bindgen_test_layout_CUDA_MEMCPY3D_PEER_st() {
    assert_eq!(::std::mem::size_of::<CUDA_MEMCPY3D_PEER_st>(), 200usize);
    assert_eq!(::std::mem::align_of::<CUDA_MEMCPY3D_PEER_st>(), 8usize);
}
#[test]
fn bindgen_test_layout_CUDA_ARRAY_DESCRIPTOR_st() {
    assert_eq!(::std::mem::size_of::<CUDA_ARRAY_DESCRIPTOR_st>(), 24usize);
    assert_eq!(::std::mem::align_of::<CUDA_ARRAY_DESCRIPTOR_st>(), 8usize);
}
#[test]
fn bindgen_test_layout_CUDA_ARRAY3D_DESCRIPTOR_st() {
    assert_eq!(::std::mem::size_of::<CUDA_ARRAY3D_DESCRIPTOR_st>(), 40usize);
    assert_eq!(::std::mem::align_of::<CUDA_ARRAY3D_DESCRIPTOR_st>(), 8usize);
}
#[test]
fn bindgen_test_layout_CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1() {
    assert_eq!(
        ::std::mem::size_of::<CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1>(),
        8usize
    );
    assert_eq!(
        ::std::mem::align_of::<CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1>(),
        8usize
    );
}
#[test]
fn bindgen_test_layout_CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_2() {
    assert_eq!(
        ::std::mem::size_of::<CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_2>(),
        8usize
    );
    assert_eq!(
        ::std::mem::align_of::<CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_2>(),
        8usize
    );
}
#[test]
fn bindgen_test_layout_CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3() {
    assert_eq!(
        ::std::mem::size_of::<CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3>(),
        24usize
    );
    assert_eq!(
        ::std::mem::align_of::<CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3>(),
        8usize
    );
}
#[test]
fn bindgen_test_layout_CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4() {
    assert_eq!(
        ::std::mem::size_of::<CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4>(),
        40usize
    );
    assert_eq!(
        ::std::mem::align_of::<CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4>(),
        8usize
    );
}
#[test]
fn bindgen_test_layout_CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_5() {
    assert_eq!(
        ::std::mem::size_of::<CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_5>(),
        128usize
    );
    assert_eq!(
        ::std::mem::align_of::<CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_5>(),
        4usize
    );
}
#[test]
fn bindgen_test_layout_CUDA_RESOURCE_DESC_st__bindgen_ty_1() {
    assert_eq!(
        ::std::mem::size_of::<CUDA_RESOURCE_DESC_st__bindgen_ty_1>(),
        128usize
    );
    assert_eq!(
        ::std::mem::align_of::<CUDA_RESOURCE_DESC_st__bindgen_ty_1>(),
        8usize
    );
}
#[test]
fn bindgen_test_layout_CUDA_RESOURCE_DESC_st() {
    assert_eq!(::std::mem::size_of::<CUDA_RESOURCE_DESC_st>(), 144usize);
    assert_eq!(::std::mem::align_of::<CUDA_RESOURCE_DESC_st>(), 8usize);
}
#[test]
fn bindgen_test_layout_CUDA_TEXTURE_DESC_st() {
    assert_eq!(::std::mem::size_of::<CUDA_TEXTURE_DESC_st>(), 104usize);
    assert_eq!(::std::mem::align_of::<CUDA_TEXTURE_DESC_st>(), 4usize);
}
#[test]
fn bindgen_test_layout_CUDA_RESOURCE_VIEW_DESC_st() {
    assert_eq!(
        ::std::mem::size_of::<CUDA_RESOURCE_VIEW_DESC_st>(),
        112usize
    );
    assert_eq!(::std::mem::align_of::<CUDA_RESOURCE_VIEW_DESC_st>(), 8usize);
}
#[test]
fn bindgen_test_layout_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st() {
    assert_eq!(
        ::std::mem::size_of::<CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st>(),
        16usize
    );
    assert_eq!(
        ::std::mem::align_of::<CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st>(),
        8usize
    );
}
