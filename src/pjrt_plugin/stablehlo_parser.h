#include <cstddef>
#include <string>

namespace xla_mpi {
    struct ParsedModule{
        bool ok();
    };

ParsedModule parseStableHLOBytecode(const char* data, size_t size);

ParsedModule parseStableHLOText(const std::string& text);

}
