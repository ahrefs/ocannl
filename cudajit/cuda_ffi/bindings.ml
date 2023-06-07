open Ctypes
(* open Bindings_types *)

module Functions (F : Ctypes.FOREIGN) = struct
  module E = Types_generated
  let cu_init = F.foreign "cuInit" F.(int @-> returning E.cu_result)
  (* let cu_device_get = F.foreign "cuDeviceGet"  *)
end