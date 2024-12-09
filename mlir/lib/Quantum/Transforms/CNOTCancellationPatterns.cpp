#define DEBUG_TYPE "cnot-cancellation"

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

struct CNOTCancellationRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Rewriting the following operation:\n" << op << "\n");
        auto loc = op.getLoc();
        StringRef opGateName = op.getGateName();
        if (opGateName != "CNOT")
            return failure();
        
        VerifyParentGateAndNameAnalysis vpga(op);
        if (!vpga.getVerifierResult())
            return failure();
        auto parentOp = cast<CustomOp>(op.getInQubits()[0].getDefiningOp());
        StringRef parentGateName = parentOp.getGateName();
        if (parentGateName != "CNOT")
            return failure();

        dbgs() << "CNOT cancellation pattern matched\n";
        // Replace uses
        std::vector<mlir::Value> originalQubits = parentOp.getQubitOperands();

        rewriter.replaceOp(op, originalQubits);
        return success();
    }
};
} // namespace

namespace catalyst {
namespace quantum {
void populateCNOTCancellationPatterns(mlir::RewritePatternSet &patterns)
{
    patterns.add<CNOTCancellationRewritePattern>(patterns.getContext());
}
} // namespace quantum
} // namespace catalyst
