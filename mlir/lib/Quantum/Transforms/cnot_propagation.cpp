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
        LLVM_DEBUG(dbgs() << "cnot propagation pass"
                          << "\n");

        Operation *module = getOperation();
        Operation *targetfunc;

        WalkResult result = module->walk([&](func::FuncOp op) {
            StringRef funcName = op.getSymName();

            if (funcName != FuncNameOpt) {
                // not the function to run the pass on, visit the next function
                return WalkResult::advance();
            }
            targetfunc = op;
            return WalkResult::interrupt();
        });

        if (!result.wasInterrupted()) {
            // Never met a target function
            // Do nothing and exit!
            return;
        }

        RewritePatternSet patterns(&getContext());
        populateCNOTPropagationPatterns(patterns);

        if (failed(applyPatternsAndFoldGreedily(targetfunc, std::move(patterns)))) {
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
