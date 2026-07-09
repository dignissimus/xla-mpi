#include <cstddef>
#include <string>

namespace xla_mpi {
    struct ParsedModule{
        std::vector<std::string> unsupported_ops;
        std::unique_ptr<mlir::MLIRContext> context;
        mlir::OwningOpRef<mlir::ModuleOp> module;
        mlir::func::FuncOp entry_func;
        bool ok();
    };

ParsedModule parseStableHLOBytecode(const char* data, size_t size);

ParsedModule parseStableHLOText(const std::string& text);

}
