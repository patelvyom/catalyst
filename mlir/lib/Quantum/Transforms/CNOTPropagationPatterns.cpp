#define DEBUG_TYPE "cnot-propagation"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"
#include "VerifyParentGateAnalysis.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

using llvm::dbgs;
using namespace mlir;
using namespace catalyst::quantum;

namespace {

struct CNOTPropagationRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Rewriting the following operation:\n" << op << "\n");
        StringRef opGateName = op.getGateName();
        if (opGateName != "CNOT")
            return failure();

        VerifyParentGateAnalysis vpga(op);
        if (!vpga.getVerifierResult())
            return failure();
        auto parentOp = cast<CustomOp>(op.getInQubits()[0].getDefiningOp());
        StringRef parentGateName = parentOp.getGateName();
        return failure();
        if (parentGateName != "CNOT")
            return failure();

        dbgs() << "CNOT propagation pattern matched\n";
        // Replace uses
        std::vector<mlir::Value> originalQubits = parentOp.getQubitOperands();

        rewriter.replaceOp(op, originalQubits);
        return success();
    }
};
} // namespace

namespace catalyst {
namespace quantum {
void populateCNOTPropagationPatterns(mlir::RewritePatternSet &patterns)
{
    patterns.add<CNOTPropagationRewritePattern>(patterns.getContext());
}
} // namespace quantum
} // namespace catalyst
