// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-pass-failure{gen-diagnostics}))' -verify-diagnostics -mlir-pass-eager-termination=false

// Test that multiple errors are reported when eager-termination is false.
// expected-error@below {{illegal operation}}
func.func @TestAlwaysIllegalOperationPass1() {
  return
}

// expected-error@below {{illegal operation}}
func.func @TestAlwaysIllegalOperationPass2() {
  return
}

// expected-error@below {{illegal operation}}
func.func @TestAlwaysIllegalOperationPass3() {
  return
}
