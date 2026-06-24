namespace xla_mpi {
    struct ParsedModule {
        bool ok() {
           return false;
        } 
    }

    ParsedModule parseStableHLOBytecode(char *code, size_t code_size) {
        return ParsedModule{};
    }


    ParsedModule parseStableHLOText(char *code, size_t code_size) {
        return ParsedModule{};
    }
};
