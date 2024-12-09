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

// struct MergeRotationsRewritePattern : public mlir::OpRewritePattern<CustomOp> {
//     using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

//     mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
//     {
//         LLVM_DEBUG(dbgs() << "Simplifying the following operation:\n" << op << "\n");
//         auto loc = op.getLoc();
//         StringRef opGateName = op.getGateName();
//         if (!rotationsSet.contains(opGateName))
//             return failure();
//         ValueRange inQubits = op.getInQubits();
//         auto parentOp = dyn_cast_or_null<CustomOp>(inQubits[0].getDefiningOp());

//         VerifyParentGateAndNameAnalysis vpga(op);
//         if (!vpga.getVerifierResult()) {
//             return failure();
//         }

//         TypeRange outQubitsTypes = op.getOutQubits().getTypes();
//         TypeRange outQubitsCtrlTypes = op.getOutCtrlQubits().getTypes();
//         ValueRange parentInQubits = parentOp.getInQubits();
//         ValueRange parentInCtrlQubits = parentOp.getInCtrlQubits();
//         ValueRange parentInCtrlValues = parentOp.getInCtrlValues();

//         auto parentParams = parentOp.getParams();
//         auto params = op.getParams();
//         SmallVector<mlir::Value> sumParams;
//         for (auto [param, parentParam] : llvm::zip(params, parentParams)) {
//             mlir::Value sumParam =
//                 rewriter.create<arith::AddFOp>(loc, parentParam, param).getResult();
//             sumParams.push_back(sumParam);
//         };
//         auto mergeOp = rewriter.create<CustomOp>(loc, outQubitsTypes, outQubitsCtrlTypes, sumParams,
//                                                  parentInQubits, opGateName, nullptr,
//                                                  parentInCtrlQubits, parentInCtrlValues);

//         op.replaceAllUsesWith(mergeOp);

//         return success();
//     }
// };

struct HadamardConjugationRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Rewriting the following operation:\n" << op << "\n");
        auto loc = op.getLoc();
        StringRef opGateName = op.getGateName();
        dbgs() << "opGateName: " << opGateName << "\n";
        if (opGateName != "Hadamard")
            return failure();
        auto parentOp = dyn_cast_or_null<CustomOp>(op.getInQubits()[0].getDefiningOp());
        if (!parentOp)
            return failure();
        StringRef parentGateName = parentOp.getGateName();
        dbgs() << "parentGateName: " << parentGateName << "\n";
        if (parentGateName != "PauliX")
            return failure();

        auto grandparentOp = dyn_cast_or_null<CustomOp>(parentOp.getInQubits()[0].getDefiningOp());
        if (!grandparentOp)
            return failure();
        StringRef grandparentGateName = grandparentOp.getGateName();
        dbgs() << "grandparentGateName: " << grandparentGateName << "\n";
        if (grandparentGateName != "Hadamard")
            return failure();

        dbgs() << "Here we should conjugate the Hadamard gate" << "\n";
        return failure(); // TODO: Implement the rewrite
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
