(* The CUDA arm of tools/vendor_idents.ml (gh-ocannl-1338): the backend's own probe, its include
   tree, and its host capabilities. Linked only on a box with cudajit (type-checked everywhere
   through arrayjit/vendor_stubs), so it holds the vendor calls and nothing else; the harvest itself
   lives in vendor_idents.ml. *)

open Base

let built = true
let probe () = Cuda_backend.Impl.routine_name_probe ()

let include_dirs () =
  List.filter_map (Cuda_backend.cuda_include_options ()) ~f:(String.chop_prefix ~prefix:"-I")

let capabilities () = Cuda_backend.host_capabilities ()
