type snapshot = unit

let unavailable () = failwith "CUDA pool backing probes require cudajit and NVRTC"
let read = unavailable
let report ?baseline:_ _ () = unavailable ()
let after_first_exec_destroy _ = unavailable ()
let streams = unavailable
