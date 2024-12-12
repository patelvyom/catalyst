#define DEBUG_TYPE "hermitian-cancellation"

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

static const mlir::StringSet<> HermitianGates = {"Hadamard", "PauliX", "PauliY", "PauliZ"};
namespace {
    
struct HermitianCancellationRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Rewriting the following operation:\n" << op << "\n");
        StringRef opGateName = op.getGateName();
        if (!HermitianGates.contains(opGateName))
            return failure();

        VerifyParentGateAndNameAnalysis vpga(op);
        if (!vpga.getVerifierResult())
            return failure();
        auto parentOp = cast<CustomOp>(op.getInQubits()[0].getDefiningOp());
        StringRef parentGateName = parentOp.getGateName();
        if (parentGateName != opGateName)
            return failure();

        dbgs() << "hermitian cancellation pattern matched\n";
        std::vector<mlir::Value> originalQubits = parentOp.getQubitOperands();
        rewriter.replaceOp(op, originalQubits);
        return success();
    }
};
} // namespace

namespace catalyst {
namespace quantum {
void populateHermitianCancellationPatterns(mlir::RewritePatternSet &patterns)
{
    patterns.add<HermitianCancellationRewritePattern>(patterns.getContext());
}
} // namespace quantum
} // namespace catalyst
