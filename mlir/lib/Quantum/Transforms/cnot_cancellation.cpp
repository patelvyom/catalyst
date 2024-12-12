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
        LLVM_DEBUG(dbgs() << "cnot cancellation pass" << "\n");

        Operation *module = getOperation();

        RewritePatternSet patterns(&getContext());
        populateCNOTCancellationPatterns(patterns);

        if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
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
