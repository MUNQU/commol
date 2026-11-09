//! The memory of compiled code: the JIT maps it, and never makes the heap
//! executable, so a formula compiles under a policy that forbids an
//! executable heap (SELinux `execheap`, the default of Fedora and RHEL).
//!
//! The test runs on the main thread (`harness = false` in `Cargo.toml`):
//! there glibc serves the allocator from the heap of `brk`, the `[heap]` of
//! `/proc/self/maps`, while the arena of another thread is a mapping of its
//! own.

#[cfg(target_os = "linux")]
fn main() {
    use commol_core::math_expression::JITCompiler;

    let function = JITCompiler::new()
        .compile("k1 + 1")
        .expect("the formula compiles");
    assert_eq!(function.call_with_buffer(&[2.5]).expect("it runs"), 3.5);

    let maps = std::fs::read_to_string("/proc/self/maps").expect("the maps of the process");
    let executable_heap: Vec<&str> = maps
        .lines()
        .filter(|line| line.ends_with("[heap]"))
        .filter(|line| {
            line.split_whitespace()
                .nth(1)
                .is_some_and(|p| p.contains('x'))
        })
        .collect();
    assert!(
        executable_heap.is_empty(),
        "the JIT made the heap executable: {executable_heap:?}"
    );
}

#[cfg(not(target_os = "linux"))]
fn main() {}
