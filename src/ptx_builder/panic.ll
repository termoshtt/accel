; ModuleID = 'panic.ll'
target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: cold noinline noreturn nounwind
define void @_ZN4core9panicking18panic_bounds_check17h476c69b1512db11aE({ [0 x i64], { [0 x i8]*, i64 }, [0 x i32], i32, [0 x i32], i32, [0 x i32] }* noalias nocapture readonly dereferenceable(24), i64, i64) unnamed_addr #6 {
  unreachable
}
