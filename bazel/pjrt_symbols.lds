# Only export GetPjrtApi
# Otherwise we get collisions with tensorflow
VERS_1.0 {
  global:
    extern "C" {
      GetPjrtApi;
    };

  local:
    *;
};
