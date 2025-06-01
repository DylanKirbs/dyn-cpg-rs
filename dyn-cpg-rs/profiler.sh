#!/usr/bin/env bash
# Usage: ./profiler.sh [cargo args]... [-- <application args>...]

sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid; echo 0 > /proc/sys/kernel/kptr_restrict'

RUST_LOG=debug RUST_BACKTRACE=1 CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph $@
