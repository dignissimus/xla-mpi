#include "pjrt_plugin/stablehlo_parser.h"
#include <cstddef>
#include <string>
#include <memory>

namespace xla_mpi {

bool ParsedModule::ok() {
    return true;
}

ParsedModule parseStableHLOBytecode(const char* data, size_t size) {
    return ParsedModule{};
    #ifdef COMMENT
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
    #endif
}


ParsedModule parseStableHLOText(const std::string& text) {
    return ParsedModule{};
}

} // namespace xla_mpi
