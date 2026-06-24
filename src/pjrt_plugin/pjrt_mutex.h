// TODO: Might be able to remove this?
inline std::recursive_mutex& GetPjrtGlobalMutex() {
    static std::recursive_mutex mutex;
    return mutex;
}

