#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]
#![cfg(test)]

use super::cudart::*;

#[test]
fn bindgen_test_layout_cudaChannelFormatDesc() {
    assert_eq!(
        ::std::mem::size_of::<cudaChannelFormatDesc>(),
        20usize,
        concat!("Size of: ", stringify!(cudaChannelFormatDesc))
    );
    assert_eq!(
        ::std::mem::align_of::<cudaChannelFormatDesc>(),
        4usize,
        concat!("Alignment of ", stringify!(cudaChannelFormatDesc))
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaChannelFormatDesc)).x as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaChannelFormatDesc),
            "::",
            stringify!(x)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaChannelFormatDesc)).y as *const _ as usize },
        4usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaChannelFormatDesc),
            "::",
            stringify!(y)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaChannelFormatDesc)).z as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaChannelFormatDesc),
            "::",
            stringify!(z)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaChannelFormatDesc)).w as *const _ as usize },
        12usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaChannelFormatDesc),
            "::",
            stringify!(w)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaChannelFormatDesc)).f as *const _ as usize },
        16usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaChannelFormatDesc),
            "::",
            stringify!(f)
        )
    );
}
#[test]
fn bindgen_test_layout_cudaPitchedPtr() {
    assert_eq!(
        ::std::mem::size_of::<cudaPitchedPtr>(),
        32usize,
        concat!("Size of: ", stringify!(cudaPitchedPtr))
    );
    assert_eq!(
        ::std::mem::align_of::<cudaPitchedPtr>(),
        8usize,
        concat!("Alignment of ", stringify!(cudaPitchedPtr))
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaPitchedPtr)).ptr as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaPitchedPtr),
            "::",
            stringify!(ptr)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaPitchedPtr)).pitch as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaPitchedPtr),
            "::",
            stringify!(pitch)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaPitchedPtr)).xsize as *const _ as usize },
        16usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaPitchedPtr),
            "::",
            stringify!(xsize)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaPitchedPtr)).ysize as *const _ as usize },
        24usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaPitchedPtr),
            "::",
            stringify!(ysize)
        )
    );
}
#[test]
fn bindgen_test_layout_cudaExtent() {
    assert_eq!(
        ::std::mem::size_of::<cudaExtent>(),
        24usize,
        concat!("Size of: ", stringify!(cudaExtent))
    );
    assert_eq!(
        ::std::mem::align_of::<cudaExtent>(),
        8usize,
        concat!("Alignment of ", stringify!(cudaExtent))
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaExtent)).width as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaExtent),
            "::",
            stringify!(width)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaExtent)).height as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaExtent),
            "::",
            stringify!(height)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaExtent)).depth as *const _ as usize },
        16usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaExtent),
            "::",
            stringify!(depth)
        )
    );
}
#[test]
fn bindgen_test_layout_cudaPos() {
    assert_eq!(
        ::std::mem::size_of::<cudaPos>(),
        24usize,
        concat!("Size of: ", stringify!(cudaPos))
    );
    assert_eq!(
        ::std::mem::align_of::<cudaPos>(),
        8usize,
        concat!("Alignment of ", stringify!(cudaPos))
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaPos)).x as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaPos),
            "::",
            stringify!(x)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaPos)).y as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaPos),
            "::",
            stringify!(y)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaPos)).z as *const _ as usize },
        16usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaPos),
            "::",
            stringify!(z)
        )
    );
}
#[test]
fn bindgen_test_layout_cudaMemcpy3DParms() {
    assert_eq!(
        ::std::mem::size_of::<cudaMemcpy3DParms>(),
        160usize,
        concat!("Size of: ", stringify!(cudaMemcpy3DParms))
    );
    assert_eq!(
        ::std::mem::align_of::<cudaMemcpy3DParms>(),
        8usize,
        concat!("Alignment of ", stringify!(cudaMemcpy3DParms))
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaMemcpy3DParms)).srcArray as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaMemcpy3DParms),
            "::",
            stringify!(srcArray)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaMemcpy3DParms)).srcPos as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaMemcpy3DParms),
            "::",
            stringify!(srcPos)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaMemcpy3DParms)).srcPtr as *const _ as usize },
        32usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaMemcpy3DParms),
            "::",
            stringify!(srcPtr)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaMemcpy3DParms)).dstArray as *const _ as usize },
        64usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaMemcpy3DParms),
            "::",
            stringify!(dstArray)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaMemcpy3DParms)).dstPos as *const _ as usize },
        72usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaMemcpy3DParms),
            "::",
            stringify!(dstPos)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaMemcpy3DParms)).dstPtr as *const _ as usize },
        96usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaMemcpy3DParms),
            "::",
            stringify!(dstPtr)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaMemcpy3DParms)).extent as *const _ as usize },
        128usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaMemcpy3DParms),
            "::",
            stringify!(extent)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaMemcpy3DParms)).kind as *const _ as usize },
        152usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaMemcpy3DParms),
            "::",
            stringify!(kind)
        )
    );
}
#[test]
fn bindgen_test_layout_cudaMemcpy3DPeerParms() {
    assert_eq!(
        ::std::mem::size_of::<cudaMemcpy3DPeerParms>(),
        168usize,
        concat!("Size of: ", stringify!(cudaMemcpy3DPeerParms))
    );
    assert_eq!(
        ::std::mem::align_of::<cudaMemcpy3DPeerParms>(),
        8usize,
        concat!("Alignment of ", stringify!(cudaMemcpy3DPeerParms))
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaMemcpy3DPeerParms)).srcArray as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaMemcpy3DPeerParms),
            "::",
            stringify!(srcArray)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaMemcpy3DPeerParms)).srcPos as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaMemcpy3DPeerParms),
            "::",
            stringify!(srcPos)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaMemcpy3DPeerParms)).srcPtr as *const _ as usize },
        32usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaMemcpy3DPeerParms),
            "::",
            stringify!(srcPtr)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaMemcpy3DPeerParms)).srcDevice as *const _ as usize },
        64usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaMemcpy3DPeerParms),
            "::",
            stringify!(srcDevice)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaMemcpy3DPeerParms)).dstArray as *const _ as usize },
        72usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaMemcpy3DPeerParms),
            "::",
            stringify!(dstArray)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaMemcpy3DPeerParms)).dstPos as *const _ as usize },
        80usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaMemcpy3DPeerParms),
            "::",
            stringify!(dstPos)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaMemcpy3DPeerParms)).dstPtr as *const _ as usize },
        104usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaMemcpy3DPeerParms),
            "::",
            stringify!(dstPtr)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaMemcpy3DPeerParms)).dstDevice as *const _ as usize },
        136usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaMemcpy3DPeerParms),
            "::",
            stringify!(dstDevice)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaMemcpy3DPeerParms)).extent as *const _ as usize },
        144usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaMemcpy3DPeerParms),
            "::",
            stringify!(extent)
        )
    );
}
#[test]
fn bindgen_test_layout_cudaResourceDesc__bindgen_ty_1__bindgen_ty_1() {
    assert_eq!(
        ::std::mem::size_of::<cudaResourceDesc__bindgen_ty_1__bindgen_ty_1>(),
        8usize,
        concat!(
            "Size of: ",
            stringify!(cudaResourceDesc__bindgen_ty_1__bindgen_ty_1)
        )
    );
    assert_eq!(
        ::std::mem::align_of::<cudaResourceDesc__bindgen_ty_1__bindgen_ty_1>(),
        8usize,
        concat!(
            "Alignment of ",
            stringify!(cudaResourceDesc__bindgen_ty_1__bindgen_ty_1)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceDesc__bindgen_ty_1__bindgen_ty_1)).array as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceDesc__bindgen_ty_1__bindgen_ty_1),
            "::",
            stringify!(array)
        )
    );
}
#[test]
fn bindgen_test_layout_cudaResourceDesc__bindgen_ty_1__bindgen_ty_2() {
    assert_eq!(
        ::std::mem::size_of::<cudaResourceDesc__bindgen_ty_1__bindgen_ty_2>(),
        8usize,
        concat!(
            "Size of: ",
            stringify!(cudaResourceDesc__bindgen_ty_1__bindgen_ty_2)
        )
    );
    assert_eq!(
        ::std::mem::align_of::<cudaResourceDesc__bindgen_ty_1__bindgen_ty_2>(),
        8usize,
        concat!(
            "Alignment of ",
            stringify!(cudaResourceDesc__bindgen_ty_1__bindgen_ty_2)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceDesc__bindgen_ty_1__bindgen_ty_2)).mipmap as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceDesc__bindgen_ty_1__bindgen_ty_2),
            "::",
            stringify!(mipmap)
        )
    );
}
#[test]
fn bindgen_test_layout_cudaResourceDesc__bindgen_ty_1__bindgen_ty_3() {
    assert_eq!(
        ::std::mem::size_of::<cudaResourceDesc__bindgen_ty_1__bindgen_ty_3>(),
        40usize,
        concat!(
            "Size of: ",
            stringify!(cudaResourceDesc__bindgen_ty_1__bindgen_ty_3)
        )
    );
    assert_eq!(
        ::std::mem::align_of::<cudaResourceDesc__bindgen_ty_1__bindgen_ty_3>(),
        8usize,
        concat!(
            "Alignment of ",
            stringify!(cudaResourceDesc__bindgen_ty_1__bindgen_ty_3)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceDesc__bindgen_ty_1__bindgen_ty_3)).devPtr as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceDesc__bindgen_ty_1__bindgen_ty_3),
            "::",
            stringify!(devPtr)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceDesc__bindgen_ty_1__bindgen_ty_3)).desc as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceDesc__bindgen_ty_1__bindgen_ty_3),
            "::",
            stringify!(desc)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceDesc__bindgen_ty_1__bindgen_ty_3)).sizeInBytes as *const _ as usize },
        32usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceDesc__bindgen_ty_1__bindgen_ty_3),
            "::",
            stringify!(sizeInBytes)
        )
    );
}
#[test]
fn bindgen_test_layout_cudaResourceDesc__bindgen_ty_1__bindgen_ty_4() {
    assert_eq!(
        ::std::mem::size_of::<cudaResourceDesc__bindgen_ty_1__bindgen_ty_4>(),
        56usize,
        concat!(
            "Size of: ",
            stringify!(cudaResourceDesc__bindgen_ty_1__bindgen_ty_4)
        )
    );
    assert_eq!(
        ::std::mem::align_of::<cudaResourceDesc__bindgen_ty_1__bindgen_ty_4>(),
        8usize,
        concat!(
            "Alignment of ",
            stringify!(cudaResourceDesc__bindgen_ty_1__bindgen_ty_4)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceDesc__bindgen_ty_1__bindgen_ty_4)).devPtr as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceDesc__bindgen_ty_1__bindgen_ty_4),
            "::",
            stringify!(devPtr)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceDesc__bindgen_ty_1__bindgen_ty_4)).desc as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceDesc__bindgen_ty_1__bindgen_ty_4),
            "::",
            stringify!(desc)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceDesc__bindgen_ty_1__bindgen_ty_4)).width as *const _ as usize },
        32usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceDesc__bindgen_ty_1__bindgen_ty_4),
            "::",
            stringify!(width)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceDesc__bindgen_ty_1__bindgen_ty_4)).height as *const _ as usize },
        40usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceDesc__bindgen_ty_1__bindgen_ty_4),
            "::",
            stringify!(height)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceDesc__bindgen_ty_1__bindgen_ty_4)).pitchInBytes as *const _ as usize },
        48usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceDesc__bindgen_ty_1__bindgen_ty_4),
            "::",
            stringify!(pitchInBytes)
        )
    );
}
#[test]
fn bindgen_test_layout_cudaResourceDesc__bindgen_ty_1() {
    assert_eq!(
        ::std::mem::size_of::<cudaResourceDesc__bindgen_ty_1>(),
        56usize,
        concat!("Size of: ", stringify!(cudaResourceDesc__bindgen_ty_1))
    );
    assert_eq!(
        ::std::mem::align_of::<cudaResourceDesc__bindgen_ty_1>(),
        8usize,
        concat!("Alignment of ", stringify!(cudaResourceDesc__bindgen_ty_1))
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceDesc__bindgen_ty_1)).array as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceDesc__bindgen_ty_1),
            "::",
            stringify!(array)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceDesc__bindgen_ty_1)).mipmap as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceDesc__bindgen_ty_1),
            "::",
            stringify!(mipmap)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceDesc__bindgen_ty_1)).linear as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceDesc__bindgen_ty_1),
            "::",
            stringify!(linear)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceDesc__bindgen_ty_1)).pitch2D as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceDesc__bindgen_ty_1),
            "::",
            stringify!(pitch2D)
        )
    );
}
#[test]
fn bindgen_test_layout_cudaResourceDesc() {
    assert_eq!(
        ::std::mem::size_of::<cudaResourceDesc>(),
        64usize,
        concat!("Size of: ", stringify!(cudaResourceDesc))
    );
    assert_eq!(
        ::std::mem::align_of::<cudaResourceDesc>(),
        8usize,
        concat!("Alignment of ", stringify!(cudaResourceDesc))
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceDesc)).resType as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceDesc),
            "::",
            stringify!(resType)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceDesc)).res as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceDesc),
            "::",
            stringify!(res)
        )
    );
}
#[test]
fn bindgen_test_layout_cudaResourceViewDesc() {
    assert_eq!(
        ::std::mem::size_of::<cudaResourceViewDesc>(),
        48usize,
        concat!("Size of: ", stringify!(cudaResourceViewDesc))
    );
    assert_eq!(
        ::std::mem::align_of::<cudaResourceViewDesc>(),
        8usize,
        concat!("Alignment of ", stringify!(cudaResourceViewDesc))
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceViewDesc)).format as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceViewDesc),
            "::",
            stringify!(format)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceViewDesc)).width as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceViewDesc),
            "::",
            stringify!(width)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceViewDesc)).height as *const _ as usize },
        16usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceViewDesc),
            "::",
            stringify!(height)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceViewDesc)).depth as *const _ as usize },
        24usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceViewDesc),
            "::",
            stringify!(depth)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceViewDesc)).firstMipmapLevel as *const _ as usize },
        32usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceViewDesc),
            "::",
            stringify!(firstMipmapLevel)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceViewDesc)).lastMipmapLevel as *const _ as usize },
        36usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceViewDesc),
            "::",
            stringify!(lastMipmapLevel)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceViewDesc)).firstLayer as *const _ as usize },
        40usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceViewDesc),
            "::",
            stringify!(firstLayer)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaResourceViewDesc)).lastLayer as *const _ as usize },
        44usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaResourceViewDesc),
            "::",
            stringify!(lastLayer)
        )
    );
}
#[test]
fn bindgen_test_layout_cudaPointerAttributes() {
    assert_eq!(
        ::std::mem::size_of::<cudaPointerAttributes>(),
        32usize,
        concat!("Size of: ", stringify!(cudaPointerAttributes))
    );
    assert_eq!(
        ::std::mem::align_of::<cudaPointerAttributes>(),
        8usize,
        concat!("Alignment of ", stringify!(cudaPointerAttributes))
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaPointerAttributes)).memoryType as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaPointerAttributes),
            "::",
            stringify!(memoryType)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaPointerAttributes)).device as *const _ as usize },
        4usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaPointerAttributes),
            "::",
            stringify!(device)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaPointerAttributes)).devicePointer as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaPointerAttributes),
            "::",
            stringify!(devicePointer)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaPointerAttributes)).hostPointer as *const _ as usize },
        16usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaPointerAttributes),
            "::",
            stringify!(hostPointer)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaPointerAttributes)).isManaged as *const _ as usize },
        24usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaPointerAttributes),
            "::",
            stringify!(isManaged)
        )
    );
}
#[test]
fn bindgen_test_layout_cudaFuncAttributes() {
    assert_eq!(
        ::std::mem::size_of::<cudaFuncAttributes>(),
        56usize,
        concat!("Size of: ", stringify!(cudaFuncAttributes))
    );
    assert_eq!(
        ::std::mem::align_of::<cudaFuncAttributes>(),
        8usize,
        concat!("Alignment of ", stringify!(cudaFuncAttributes))
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaFuncAttributes)).sharedSizeBytes as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaFuncAttributes),
            "::",
            stringify!(sharedSizeBytes)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaFuncAttributes)).constSizeBytes as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaFuncAttributes),
            "::",
            stringify!(constSizeBytes)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaFuncAttributes)).localSizeBytes as *const _ as usize },
        16usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaFuncAttributes),
            "::",
            stringify!(localSizeBytes)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaFuncAttributes)).maxThreadsPerBlock as *const _ as usize },
        24usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaFuncAttributes),
            "::",
            stringify!(maxThreadsPerBlock)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaFuncAttributes)).numRegs as *const _ as usize },
        28usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaFuncAttributes),
            "::",
            stringify!(numRegs)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaFuncAttributes)).ptxVersion as *const _ as usize },
        32usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaFuncAttributes),
            "::",
            stringify!(ptxVersion)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaFuncAttributes)).binaryVersion as *const _ as usize },
        36usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaFuncAttributes),
            "::",
            stringify!(binaryVersion)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaFuncAttributes)).cacheModeCA as *const _ as usize },
        40usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaFuncAttributes),
            "::",
            stringify!(cacheModeCA)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaFuncAttributes)).maxDynamicSharedSizeBytes as *const _ as usize },
        44usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaFuncAttributes),
            "::",
            stringify!(maxDynamicSharedSizeBytes)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaFuncAttributes)).preferredShmemCarveout as *const _ as usize },
        48usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaFuncAttributes),
            "::",
            stringify!(preferredShmemCarveout)
        )
    );
}
#[test]
fn bindgen_test_layout_cudaDeviceProp() {
    assert_eq!(
        ::std::mem::size_of::<cudaDeviceProp>(),
        672usize,
        concat!("Size of: ", stringify!(cudaDeviceProp))
    );
    assert_eq!(
        ::std::mem::align_of::<cudaDeviceProp>(),
        8usize,
        concat!("Alignment of ", stringify!(cudaDeviceProp))
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).name as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(name)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).totalGlobalMem as *const _ as usize },
        256usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(totalGlobalMem)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).sharedMemPerBlock as *const _ as usize },
        264usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(sharedMemPerBlock)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).regsPerBlock as *const _ as usize },
        272usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(regsPerBlock)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).warpSize as *const _ as usize },
        276usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(warpSize)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).memPitch as *const _ as usize },
        280usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(memPitch)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxThreadsPerBlock as *const _ as usize },
        288usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxThreadsPerBlock)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxThreadsDim as *const _ as usize },
        292usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxThreadsDim)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxGridSize as *const _ as usize },
        304usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxGridSize)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).clockRate as *const _ as usize },
        316usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(clockRate)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).totalConstMem as *const _ as usize },
        320usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(totalConstMem)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).major as *const _ as usize },
        328usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(major)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).minor as *const _ as usize },
        332usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(minor)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).textureAlignment as *const _ as usize },
        336usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(textureAlignment)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).texturePitchAlignment as *const _ as usize },
        344usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(texturePitchAlignment)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).deviceOverlap as *const _ as usize },
        352usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(deviceOverlap)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).multiProcessorCount as *const _ as usize },
        356usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(multiProcessorCount)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).kernelExecTimeoutEnabled as *const _ as usize },
        360usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(kernelExecTimeoutEnabled)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).integrated as *const _ as usize },
        364usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(integrated)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).canMapHostMemory as *const _ as usize },
        368usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(canMapHostMemory)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).computeMode as *const _ as usize },
        372usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(computeMode)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxTexture1D as *const _ as usize },
        376usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture1D)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxTexture1DMipmap as *const _ as usize },
        380usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture1DMipmap)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxTexture1DLinear as *const _ as usize },
        384usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture1DLinear)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxTexture2D as *const _ as usize },
        388usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture2D)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxTexture2DMipmap as *const _ as usize },
        396usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture2DMipmap)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxTexture2DLinear as *const _ as usize },
        404usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture2DLinear)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxTexture2DGather as *const _ as usize },
        416usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture2DGather)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxTexture3D as *const _ as usize },
        424usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture3D)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxTexture3DAlt as *const _ as usize },
        436usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture3DAlt)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxTextureCubemap as *const _ as usize },
        448usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTextureCubemap)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxTexture1DLayered as *const _ as usize },
        452usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture1DLayered)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxTexture2DLayered as *const _ as usize },
        460usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture2DLayered)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxTextureCubemapLayered as *const _ as usize },
        472usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTextureCubemapLayered)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxSurface1D as *const _ as usize },
        480usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxSurface1D)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxSurface2D as *const _ as usize },
        484usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxSurface2D)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxSurface3D as *const _ as usize },
        492usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxSurface3D)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxSurface1DLayered as *const _ as usize },
        504usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxSurface1DLayered)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxSurface2DLayered as *const _ as usize },
        512usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxSurface2DLayered)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxSurfaceCubemap as *const _ as usize },
        524usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxSurfaceCubemap)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxSurfaceCubemapLayered as *const _ as usize },
        528usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxSurfaceCubemapLayered)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).surfaceAlignment as *const _ as usize },
        536usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(surfaceAlignment)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).concurrentKernels as *const _ as usize },
        544usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(concurrentKernels)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).ECCEnabled as *const _ as usize },
        548usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(ECCEnabled)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).pciBusID as *const _ as usize },
        552usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(pciBusID)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).pciDeviceID as *const _ as usize },
        556usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(pciDeviceID)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).pciDomainID as *const _ as usize },
        560usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(pciDomainID)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).tccDriver as *const _ as usize },
        564usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(tccDriver)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).asyncEngineCount as *const _ as usize },
        568usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(asyncEngineCount)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).unifiedAddressing as *const _ as usize },
        572usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(unifiedAddressing)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).memoryClockRate as *const _ as usize },
        576usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(memoryClockRate)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).memoryBusWidth as *const _ as usize },
        580usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(memoryBusWidth)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).l2CacheSize as *const _ as usize },
        584usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(l2CacheSize)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).maxThreadsPerMultiProcessor as *const _ as usize },
        588usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxThreadsPerMultiProcessor)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).streamPrioritiesSupported as *const _ as usize },
        592usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(streamPrioritiesSupported)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).globalL1CacheSupported as *const _ as usize },
        596usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(globalL1CacheSupported)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).localL1CacheSupported as *const _ as usize },
        600usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(localL1CacheSupported)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).sharedMemPerMultiprocessor as *const _ as usize },
        608usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(sharedMemPerMultiprocessor)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).regsPerMultiprocessor as *const _ as usize },
        616usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(regsPerMultiprocessor)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).managedMemory as *const _ as usize },
        620usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(managedMemory)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).isMultiGpuBoard as *const _ as usize },
        624usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(isMultiGpuBoard)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).multiGpuBoardGroupID as *const _ as usize },
        628usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(multiGpuBoardGroupID)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).hostNativeAtomicSupported as *const _ as usize },
        632usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(hostNativeAtomicSupported)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).singleToDoublePrecisionPerfRatio as *const _ as usize },
        636usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(singleToDoublePrecisionPerfRatio)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).pageableMemoryAccess as *const _ as usize },
        640usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(pageableMemoryAccess)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).concurrentManagedAccess as *const _ as usize },
        644usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(concurrentManagedAccess)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).computePreemptionSupported as *const _ as usize },
        648usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(computePreemptionSupported)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).canUseHostPointerForRegisteredMem as *const _ as usize },
        652usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(canUseHostPointerForRegisteredMem)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).cooperativeLaunch as *const _ as usize },
        656usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(cooperativeLaunch)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).cooperativeMultiDeviceLaunch as *const _ as usize },
        660usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(cooperativeMultiDeviceLaunch)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaDeviceProp)).sharedMemPerBlockOptin as *const _ as usize },
        664usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(sharedMemPerBlockOptin)
        )
    );
}
#[test]
fn bindgen_test_layout_cudaIpcEventHandle_st() {
    assert_eq!(
        ::std::mem::size_of::<cudaIpcEventHandle_st>(),
        64usize,
        concat!("Size of: ", stringify!(cudaIpcEventHandle_st))
    );
    assert_eq!(
        ::std::mem::align_of::<cudaIpcEventHandle_st>(),
        1usize,
        concat!("Alignment of ", stringify!(cudaIpcEventHandle_st))
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaIpcEventHandle_st)).reserved as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaIpcEventHandle_st),
            "::",
            stringify!(reserved)
        )
    );
}
#[test]
fn bindgen_test_layout_cudaIpcMemHandle_st() {
    assert_eq!(
        ::std::mem::size_of::<cudaIpcMemHandle_st>(),
        64usize,
        concat!("Size of: ", stringify!(cudaIpcMemHandle_st))
    );
    assert_eq!(
        ::std::mem::align_of::<cudaIpcMemHandle_st>(),
        1usize,
        concat!("Alignment of ", stringify!(cudaIpcMemHandle_st))
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaIpcMemHandle_st)).reserved as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaIpcMemHandle_st),
            "::",
            stringify!(reserved)
        )
    );
}
#[test]
fn bindgen_test_layout_cudaLaunchParams() {
    assert_eq!(
        ::std::mem::size_of::<cudaLaunchParams>(),
        56usize,
        concat!("Size of: ", stringify!(cudaLaunchParams))
    );
    assert_eq!(
        ::std::mem::align_of::<cudaLaunchParams>(),
        8usize,
        concat!("Alignment of ", stringify!(cudaLaunchParams))
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaLaunchParams)).func as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaLaunchParams),
            "::",
            stringify!(func)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaLaunchParams)).gridDim as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaLaunchParams),
            "::",
            stringify!(gridDim)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaLaunchParams)).blockDim as *const _ as usize },
        20usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaLaunchParams),
            "::",
            stringify!(blockDim)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaLaunchParams)).args as *const _ as usize },
        32usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaLaunchParams),
            "::",
            stringify!(args)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaLaunchParams)).sharedMem as *const _ as usize },
        40usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaLaunchParams),
            "::",
            stringify!(sharedMem)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaLaunchParams)).stream as *const _ as usize },
        48usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaLaunchParams),
            "::",
            stringify!(stream)
        )
    );
}
#[test]
fn bindgen_test_layout_surfaceReference() {
    assert_eq!(
        ::std::mem::size_of::<surfaceReference>(),
        20usize,
        concat!("Size of: ", stringify!(surfaceReference))
    );
    assert_eq!(
        ::std::mem::align_of::<surfaceReference>(),
        4usize,
        concat!("Alignment of ", stringify!(surfaceReference))
    );
    assert_eq!(
        unsafe { &(*(0 as *const surfaceReference)).channelDesc as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(surfaceReference),
            "::",
            stringify!(channelDesc)
        )
    );
}
#[test]
fn bindgen_test_layout_textureReference() {
    assert_eq!(
        ::std::mem::size_of::<textureReference>(),
        124usize,
        concat!("Size of: ", stringify!(textureReference))
    );
    assert_eq!(
        ::std::mem::align_of::<textureReference>(),
        4usize,
        concat!("Alignment of ", stringify!(textureReference))
    );
    assert_eq!(
        unsafe { &(*(0 as *const textureReference)).normalized as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(textureReference),
            "::",
            stringify!(normalized)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const textureReference)).filterMode as *const _ as usize },
        4usize,
        concat!(
            "Offset of field: ",
            stringify!(textureReference),
            "::",
            stringify!(filterMode)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const textureReference)).addressMode as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(textureReference),
            "::",
            stringify!(addressMode)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const textureReference)).channelDesc as *const _ as usize },
        20usize,
        concat!(
            "Offset of field: ",
            stringify!(textureReference),
            "::",
            stringify!(channelDesc)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const textureReference)).sRGB as *const _ as usize },
        40usize,
        concat!(
            "Offset of field: ",
            stringify!(textureReference),
            "::",
            stringify!(sRGB)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const textureReference)).maxAnisotropy as *const _ as usize },
        44usize,
        concat!(
            "Offset of field: ",
            stringify!(textureReference),
            "::",
            stringify!(maxAnisotropy)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const textureReference)).mipmapFilterMode as *const _ as usize },
        48usize,
        concat!(
            "Offset of field: ",
            stringify!(textureReference),
            "::",
            stringify!(mipmapFilterMode)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const textureReference)).mipmapLevelBias as *const _ as usize },
        52usize,
        concat!(
            "Offset of field: ",
            stringify!(textureReference),
            "::",
            stringify!(mipmapLevelBias)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const textureReference)).minMipmapLevelClamp as *const _ as usize },
        56usize,
        concat!(
            "Offset of field: ",
            stringify!(textureReference),
            "::",
            stringify!(minMipmapLevelClamp)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const textureReference)).maxMipmapLevelClamp as *const _ as usize },
        60usize,
        concat!(
            "Offset of field: ",
            stringify!(textureReference),
            "::",
            stringify!(maxMipmapLevelClamp)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const textureReference)).__cudaReserved as *const _ as usize },
        64usize,
        concat!(
            "Offset of field: ",
            stringify!(textureReference),
            "::",
            stringify!(__cudaReserved)
        )
    );
}
#[test]
fn bindgen_test_layout_cudaTextureDesc() {
    assert_eq!(
        ::std::mem::size_of::<cudaTextureDesc>(),
        64usize,
        concat!("Size of: ", stringify!(cudaTextureDesc))
    );
    assert_eq!(
        ::std::mem::align_of::<cudaTextureDesc>(),
        4usize,
        concat!("Alignment of ", stringify!(cudaTextureDesc))
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaTextureDesc)).addressMode as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaTextureDesc),
            "::",
            stringify!(addressMode)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaTextureDesc)).filterMode as *const _ as usize },
        12usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaTextureDesc),
            "::",
            stringify!(filterMode)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaTextureDesc)).readMode as *const _ as usize },
        16usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaTextureDesc),
            "::",
            stringify!(readMode)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaTextureDesc)).sRGB as *const _ as usize },
        20usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaTextureDesc),
            "::",
            stringify!(sRGB)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaTextureDesc)).borderColor as *const _ as usize },
        24usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaTextureDesc),
            "::",
            stringify!(borderColor)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaTextureDesc)).normalizedCoords as *const _ as usize },
        40usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaTextureDesc),
            "::",
            stringify!(normalizedCoords)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaTextureDesc)).maxAnisotropy as *const _ as usize },
        44usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaTextureDesc),
            "::",
            stringify!(maxAnisotropy)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaTextureDesc)).mipmapFilterMode as *const _ as usize },
        48usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaTextureDesc),
            "::",
            stringify!(mipmapFilterMode)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaTextureDesc)).mipmapLevelBias as *const _ as usize },
        52usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaTextureDesc),
            "::",
            stringify!(mipmapLevelBias)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaTextureDesc)).minMipmapLevelClamp as *const _ as usize },
        56usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaTextureDesc),
            "::",
            stringify!(minMipmapLevelClamp)
        )
    );
    assert_eq!(
        unsafe { &(*(0 as *const cudaTextureDesc)).maxMipmapLevelClamp as *const _ as usize },
        60usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaTextureDesc),
            "::",
            stringify!(maxMipmapLevelClamp)
        )
    );
}
