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

static const mlir::StringSet<> PropagationOps = {"PauliX", "PauliY", "PauliZ"};
namespace {

struct CNOTPropagationRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Rewriting the following operation:\n" << op << "\n");
        StringRef opGateName = op.getGateName();
        if (opGateName != "CNOT")
            return failure();

        auto parentOp = cast<CustomOp>(op.getInQubits()[0].getDefiningOp());
        StringRef parentOpGateName = parentOp.getGateName();
        if (!PropagationOps.contains(parentOpGateName))
            return failure();

        if (op.getInQubits().size() != 2 || parentOp.getOutQubits().size() != 1)
            return failure();

        mlir::Value inCtrlQubit = op.getInQubits().front();
        mlir::Value inTargQubit = op.getInQubits().back();
        mlir::Value parentOutQubit = parentOp.getOutQubits().front();

        bool foundCtrlMatch = (inCtrlQubit == parentOutQubit) ? true : false;
        bool foundNonCtrlMatch = (inTargQubit == parentOutQubit) ? true : false;

        // If neither control nor target match with parent's output, pattern doesn't match
        if (!foundNonCtrlMatch && !foundCtrlMatch)
            return failure();

        dbgs() << "CNOT propagation pattern matched\n";
        auto opLoc = op.getLoc();
        if (parentOpGateName == "PauliX") {
            if (foundCtrlMatch) {
                auto cnotOp = cast<quantum::CustomOp>(op);
                Operation *definingOp = cnotOp.getInQubits().front().getDefiningOp();
                auto xOp = cast<quantum::CustomOp>(definingOp);

                mlir::Location opLoc = op->getLoc();
                // Create new CNOT operation with original non-X input
                SmallVector<mlir::Value> cnotInQubits;
                cnotInQubits.push_back(xOp.getInQubits().front()); // Use input to X gate
                cnotInQubits.push_back(cnotOp.getInQubits().back());

                auto newCnotOp = rewriter.create<quantum::CustomOp>(
                                 opLoc,
                                 cnotOp.getOutQubits().getTypes(),
                                 ValueRange{},
                                 cnotOp.getParams(),
                                 cnotInQubits,
                                 "CNOT",
                                 nullptr,
                                 ValueRange{},
                                 ValueRange{});

                // Create X gates operating on CNOT outputs
                auto xOp1 = rewriter.create<quantum::CustomOp>(
                            opLoc,
                            newCnotOp.getOutQubits().front().getType(),
                            ValueRange{},
                            xOp.getParams(),
                            newCnotOp.getOutQubits().front(),
                            "PauliX",
                            nullptr,
                            ValueRange{},
                            ValueRange{});
                auto xOp2 = rewriter.create<quantum::CustomOp>(
                            opLoc,
                            newCnotOp.getOutQubits().back().getType(),
                            ValueRange{},
                            xOp.getParams(),
                            newCnotOp.getOutQubits().back(),
                            "PauliX",
                            nullptr,
                            ValueRange{},
                            ValueRange{});

                // SmallVector<mlir::Value> newOp;
                // newOp.push_back(newCnotOp.getOutQubits().front());
                // newOp.push_back(xOp1.getOutQubits().front());
                // newOp.push_back(xOp2.getOutQubits().front());
                rewriter.replaceOp(cnotOp, newCnotOp);
                // rewriter.eraseOp(xOp); // Remove original X gate
                return success();
            }
            else {
                return failure();
            }
        }
        else if (parentOpGateName == "PauliZ") {
            if (foundNonCtrlMatch) { // CNOT propagates Z from target to control
                return failure();
            }
            else {
                return failure();
            }
        }

        return failure();
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
