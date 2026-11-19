(* The HIP arm of tools/vendor_idents.ml (gh-ocannl-1338): the backend's own probe, its include
   tree, and its host capabilities. Linked only on a box with hipjit (type-checked everywhere
   through arrayjit/vendor_stubs), so it holds the vendor calls and nothing else; the harvest itself
   lives in vendor_idents.ml.

   hiprtc carries built-in headers on Linux. The harvest reads the corresponding installed SDK
   sources and rocWMMA through the backend's include paths. *)

open Base

let built = true

let probes () =
  [
    ("ordinary HIP", Hip_backend.Impl.routine_name_probe ~with_rocwmma:false ());
    ("HIP with rocWMMA", Hip_backend.Impl.routine_name_probe ~with_rocwmma:true ());
  ]

let include_dirs () =
  List.filter_map (Hip_backend.hip_probe_include_options ()) ~f:(String.chop_prefix ~prefix:"-I")

let capabilities () = Hip_backend.host_capabilities ()
