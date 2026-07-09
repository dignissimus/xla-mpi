#include <cstddef>
#include <string>
#include <memory>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"


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
