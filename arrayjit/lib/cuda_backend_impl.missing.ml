module Impl = Lowered_backend_missing.Missing (struct
  let name = "cuda"
end)

let host_capabilities () = []
