#define DEBUG_TYPE "cnot-propagation"

#include <memory>
#include <vector>

#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_CNOTPROPAGATIONPASS
#define GEN_PASS_DECL_CNOTPROPAGATIONPASS
#include "Quantum/Transforms/Passes.h.inc"

struct CNOTPropagationPass
    : impl::CNOTPropagationPassBase<CNOTPropagationPass> {
    using CNOTPropagationPassBase::CNOTPropagationPassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(dbgs() << "cnot propagation pass" << "\n");

        Operation *module = getOperation();
        
        RewritePatternSet patterns(&getContext());
        populateCNOTPropagationPatterns(patterns);

        if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum

std::unique_ptr<Pass> createCNOTPropagationPass()
{
    return std::make_unique<quantum::CNOTPropagationPass>();
}

} // namespace catalyst
