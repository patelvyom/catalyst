#define DEBUG_TYPE "cnot-cancellation"

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

#define GEN_PASS_DEF_CNOTCANCELLATIONPASS
#define GEN_PASS_DECL_CNOTCANCELLATIONPASS
#include "Quantum/Transforms/Passes.h.inc"

struct CNOTCancellationPass
    : impl::CNOTCancellationPassBase<CNOTCancellationPass> {
    using CNOTCancellationPassBase::CNOTCancellationPassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(dbgs() << "cnot cancellation pass"
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
        populateCNOTCancellationPatterns(patterns);

        if (failed(applyPatternsAndFoldGreedily(targetfunc, std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum

std::unique_ptr<Pass> createCNOTCancellationPass()
{
    return std::make_unique<quantum::CNOTCancellationPass>();
}

} // namespace catalyst
