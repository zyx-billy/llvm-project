// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-pass-failure{gen-diagnostics}))' -verify-diagnostics -mlir-pass-eager-termination -mlir-disable-threading

// Test that only the first error is reported when eager-termination is true.
// Multi-threading is disabled so that error reporting is deterministic.
// expected-error@below {{illegal operation}}
func.func @TestAlwaysIllegalOperationPass1() {
  return
}

func.func @TestAlwaysIllegalOperationPass2() {
  return
}

func.func @TestAlwaysIllegalOperationPass3() {
  return
}
