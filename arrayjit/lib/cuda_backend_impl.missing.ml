module Fresh =
struct
  include Lowered_backend_missing.Missing (struct
    let name = "cuda"
  end)
end
