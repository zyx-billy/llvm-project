//===- LLVMInterfaces.h - LLVM Interfaces -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines op interfaces for the LLVM dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMINTERFACES_H_
#define MLIR_DIALECT_LLVMIR_LLVMINTERFACES_H_

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"

namespace mlir {
namespace LLVM {
namespace detail {

/// Verifies the access groups attribute of memory operations that implement the
/// access group interface.
LogicalResult verifyAccessGroupOpInterface(Operation *op);

/// Verifies the alias analysis attributes of memory operations that implement
/// the alias analysis interface.
LogicalResult verifyAliasAnalysisOpInterface(Operation *op);

/// Caching verifier for DINodeAttrs that may contain recursive DI types.
class DIRecursiveTypeVerifier {
public:
  DIRecursiveTypeVerifier(Location errorLoc) : errorLoc(errorLoc) {}

  /// Verifies that DIRecursiveTypeAttr usages inside other attributes are
  /// legal.
  /// 1. There must be no unbound recursive self-references.
  /// 2. There must not be any ambiguous recursive IDs.
  /// This is an expensive check that walks the entire attribute tree.
  LogicalResult verify(DINodeAttr attr);

private:
  /// Recursive verification helper.
  /// `context` contains the set of recIds that are currently in scope when
  /// evaluating `attr`.
  /// `unboundSelfRefs` contains the set of recIds that are contained in `attr`
  /// upon return.
  LogicalResult
  verifyDIRecursiveTypesWithContext(Attribute attr,
                                    DenseSet<DistinctAttr> &context,
                                    DenseSet<DistinctAttr> &unboundSelfRefs);

  DenseSet<Attribute> knownLegals;

  /// The location to use for reporting errors.
  Location errorLoc;
};

/// Prune nested recursive declarations.
DINodeAttr pruneDIRecursiveTypes(DINodeAttr attr);

} // namespace detail
} // namespace LLVM
} // namespace mlir

#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h.inc"

#endif // MLIR_DIALECT_LLVMIR_LLVMINTERFACES_H_
