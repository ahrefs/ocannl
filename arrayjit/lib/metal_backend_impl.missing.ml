module Fresh =
struct
  include Lowered_backend_missing.Missing (struct
    let name = "metal"
  end)
end
