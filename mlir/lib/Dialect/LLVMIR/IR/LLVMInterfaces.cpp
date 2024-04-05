//===- LLVMInterfaces.cpp - LLVM Interfaces ---------------------*- C++ -*-===//
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

#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace mlir::LLVM;

/// Verifies that all elements of `array` are instances of `Attr`.
template <class AttrT>
static LogicalResult isArrayOf(Operation *op, ArrayAttr array) {
  for (Attribute iter : array)
    if (!isa<AttrT>(iter))
      return op->emitOpError("expected op to return array of ")
             << AttrT::getMnemonic() << " attributes";
  return success();
}

//===----------------------------------------------------------------------===//
// AccessGroupOpInterface
//===----------------------------------------------------------------------===//

LogicalResult mlir::LLVM::detail::verifyAccessGroupOpInterface(Operation *op) {
  auto iface = cast<AccessGroupOpInterface>(op);
  ArrayAttr accessGroups = iface.getAccessGroupsOrNull();
  if (!accessGroups)
    return success();

  return isArrayOf<AccessGroupAttr>(op, accessGroups);
}

//===----------------------------------------------------------------------===//
// AliasAnalysisOpInterface
//===----------------------------------------------------------------------===//

LogicalResult
mlir::LLVM::detail::verifyAliasAnalysisOpInterface(Operation *op) {
  auto iface = cast<AliasAnalysisOpInterface>(op);

  if (auto aliasScopes = iface.getAliasScopesOrNull())
    if (failed(isArrayOf<AliasScopeAttr>(op, aliasScopes)))
      return failure();

  if (auto noAliasScopes = iface.getNoAliasScopesOrNull())
    if (failed(isArrayOf<AliasScopeAttr>(op, noAliasScopes)))
      return failure();

  ArrayAttr tags = iface.getTBAATagsOrNull();
  if (!tags)
    return success();

  return isArrayOf<TBAATagAttr>(op, tags);
}

SmallVector<Value> mlir::LLVM::AtomicCmpXchgOp::getAccessedOperands() {
  return {getPtr()};
}

SmallVector<Value> mlir::LLVM::AtomicRMWOp::getAccessedOperands() {
  return {getPtr()};
}

SmallVector<Value> mlir::LLVM::LoadOp::getAccessedOperands() {
  return {getAddr()};
}

SmallVector<Value> mlir::LLVM::StoreOp::getAccessedOperands() {
  return {getAddr()};
}

SmallVector<Value> mlir::LLVM::MemcpyOp::getAccessedOperands() {
  return {getDst(), getSrc()};
}

SmallVector<Value> mlir::LLVM::MemcpyInlineOp::getAccessedOperands() {
  return {getDst(), getSrc()};
}

SmallVector<Value> mlir::LLVM::MemmoveOp::getAccessedOperands() {
  return {getDst(), getSrc()};
}

SmallVector<Value> mlir::LLVM::MemsetOp::getAccessedOperands() {
  return {getDst()};
}

SmallVector<Value> mlir::LLVM::CallOp::getAccessedOperands() {
  return llvm::to_vector(
      llvm::make_filter_range(getArgOperands(), [](Value arg) {
        return isa<LLVMPointerType>(arg.getType());
      }));
}

//===----------------------------------------------------------------------===//
// DIRecursiveTypeAttrInterface
//===----------------------------------------------------------------------===//
LogicalResult
mlir::LLVM::detail::DIRecursiveTypeVerifier::verify(DINodeAttr attr) {
  DenseSet<DistinctAttr> context;
  DenseSet<DistinctAttr> unboundSelfRefs;
  return verifyDIRecursiveTypesWithContext(attr, context, unboundSelfRefs);
}

LogicalResult
mlir::LLVM::detail::DIRecursiveTypeVerifier::verifyDIRecursiveTypesWithContext(
    Attribute attr, DenseSet<DistinctAttr> &context,
    DenseSet<DistinctAttr> &unboundSelfRefs) {
  if (knownLegals.contains(attr))
    return success();

  DistinctAttr recId;
  if (auto recType = dyn_cast<DIRecursiveTypeAttrInterface>(attr)) {
    if ((recId = recType.getRecId())) {
      if (recType.isRecSelf()) {
        unboundSelfRefs.insert(recId);
        if (context.contains(recId))
          return success();
        return emitError(errorLoc)
               << "Unbound recursive self-reference to " << recId;
      }

      auto [_, inserted] = context.insert(recId);
      if (!inserted)
        return emitError(errorLoc) << "Duplicate recursive ID: " << recId;
    }
  }

  bool recursivelySucceeded = true;
  attr.walkImmediateSubElements(
      [&](Attribute innerAttr) {
        recursivelySucceeded &= succeeded(verifyDIRecursiveTypesWithContext(
            innerAttr, context, unboundSelfRefs));
      },
      [&](Type innerType) {});

  if (recId) {
    context.erase(recId);
    unboundSelfRefs.erase(recId);
  }

  if (recursivelySucceeded && unboundSelfRefs.empty())
    knownLegals.insert(attr);

  return success(recursivelySucceeded);
}

#include "mlir/Dialect/LLVMIR/LLVMInterfaces.cpp.inc"
