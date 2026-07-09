#include "pjrt_plugin/stablehlo_parser.h"
#include <cstddef>
#include <string>
#include <memory>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/Serialization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/dialect/ChloOps.h"

namespace xla_mpi {

bool ParsedModule::ok() {
    return true;
}

void registerDialects(mlir::MLIRContext& context) {
    // TODO: Do I want to disable multithreading?
    context.disableMultithreading();
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::stablehlo::StablehloDialect>();
    registry.insert<mlir::vhlo::VhloDialect>();
    registry.insert<mlir::chlo::ChloDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
    context.allowUnregisteredDialects();
}

// TODO: Implement stubs
bool runInlinerPass(mlir::MLIRContext& ctx, mlir::ModuleOp& mod) { return true; }
    bool runOptimizationPasses(mlir::MLIRContext& ctx, mlir::ModuleOp& mod) { return true; }
    mlir::func::FuncOp findEntryFunction(mlir::ModuleOp& mod) { 
        return nullptr; 
    }
    std::vector<std::string> checkUnsupportedOps(mlir::ModuleOp& mod) { return {}; }

ParsedModule finalizeModule(std::unique_ptr<mlir::MLIRContext> context,
                            mlir::OwningOpRef<mlir::ModuleOp> module) {
    ParsedModule result;

    if (!module) {
        return result;
    }

    // TODO: Why inline?
    // Inline all func.call operations
    if (!runInlinerPass(*context, *module)) {
        return result;
    }

    if (!runOptimizationPasses(*context, *module)) {
        return result;
    }

    // TODO: Dump debug info

    mlir::func::FuncOp entry = findEntryFunction(*module);
    if (!entry) {
        return result;
    }

    result.unsupported_ops = checkUnsupportedOps(*module);

    // Transfer ownership
    result.context = std::move(context);
    result.module = std::move(module);
    result.entry_func = entry;

    return result;
}


ParsedModule parseStableHLOBytecode(const char* data, size_t size) {
    return ParsedModule{};
    auto context = std::make_unique<mlir::MLIRContext>();
    registerDialects(*context);

    auto buffer = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(data, size),
                                                   /*BufferName=*/"stablehlo_bytecode",
                                                   /*RequiresNullTerminator=*/false);

    // Try to deserialize as portable artifact first
    mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
        mlir::stablehlo::deserializePortableArtifact(buffer->getBuffer(), context.get());

    // Fall back to regular bytecode reading
    if (!moduleOp) {
        llvm::SourceMgr sourceMgr;
        sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

        moduleOp = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, context.get());
    }

    return finalizeModule(std::move(context), std::move(moduleOp));
}


ParsedModule parseStableHLOText(const std::string& text) {
    return ParsedModule{};
}

} // namespace xla_mpi
