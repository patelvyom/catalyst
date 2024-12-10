#define DEBUG_TYPE "hadamard-conjugation"

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
struct HadamardConjugationRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Rewriting the following operation:\n" << op << "\n");
        StringRef opGateName = op.getGateName();
        if (opGateName != "Hadamard")
            return failure();
        auto parentOp = dyn_cast_or_null<CustomOp>(op.getInQubits()[0].getDefiningOp());
        VerifyParentGateAnalysis<CustomOp> vpga(parentOp);
        if (!vpga.getVerifierResult())
            return failure();
        StringRef parentGateName = parentOp.getGateName();
        if (parentGateName != "PauliX" && parentGateName != "PauliZ")
            return failure();

        auto grandParentOp = dyn_cast_or_null<CustomOp>(parentOp.getInQubits()[0].getDefiningOp());
        VerifyParentGateAnalysis<CustomOp> vgpga(grandParentOp);
        if (!vgpga.getVerifierResult())
            return failure();
        StringRef grandParentGateName = grandParentOp.getGateName();
        if (grandParentGateName != "Hadamard")
            return failure();

        auto loc = op.getLoc();
        dbgs() << "Found Hadamard conjugation at loc " << loc << "\n";

        // Replace current op with X. Set input of X to input of H (grandparent op) and output of X
        // to output of H (parent op).
        auto newOpParams = parentOp.getParams();
        StringRef newOpName = (parentGateName == "PauliX") ? "PauliZ" : "PauliX";
        TypeRange newOpQubitsTypes = op.getOutQubits().getTypes();
        ValueRange newOpInQubits = grandParentOp.getInQubits();

        auto mergeOp = rewriter.create<CustomOp>(loc, newOpQubitsTypes, ValueRange{}, newOpParams,
                                                 newOpInQubits, newOpName, nullptr, ValueRange{},
                                                 ValueRange{});
        op.replaceAllUsesWith(mergeOp);
        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateHadamardConjugationPatterns(RewritePatternSet &patterns)
{
    patterns.add<HadamardConjugationRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
