#define DEBUG_TYPE "hadamard-conjugation"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_HADAMARDCONJUGATIONPASS
#define GEN_PASS_DECL_HADAMARDCONJUGATIONPASS
#include "Quantum/Transforms/Passes.h.inc"

struct HadamardConjugationPass : impl::HadamardConjugationPassBase<HadamardConjugationPass> {
    using HadamardConjugationPassBase::HadamardConjugationPassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(dbgs() << "hadamard conjugation pass" << "\n");

        Operation *module = getOperation();

        RewritePatternSet patterns(&getContext());
        populateHadamardConjugationPatterns(patterns);

        if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum

std::unique_ptr<Pass> createHadamardConjugationPass()
{
    return std::make_unique<quantum::HadamardConjugationPass>();
}

} // namespace catalyst
