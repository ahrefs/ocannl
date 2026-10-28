(* The HIP arm of tools/vendor_idents.ml (gh-ocannl-1338): the backend's own probe, its include
   tree, and its host capabilities. Compiled only on a box with hipjit, so it holds the vendor calls
   and nothing else; the harvest itself lives in vendor_idents.ml.

   hiprtc carries built-in headers on Linux, where [Hip_backend.hip_include_options] is then empty:
   the tokens are read from the installed SDK tree, the same headers' sources. *)

open Base

let built = true
let probe () = Hip_backend.Impl.routine_name_probe ()

let include_dirs () =
  let from_backend =
    List.filter_map (Hip_backend.hip_include_options ()) ~f:(String.chop_prefix ~prefix:"-I")
  in
  let sdk =
    List.filter [ "/opt/rocm/include"; "/usr/include" ] ~f:(fun d ->
        Stdlib.Sys.file_exists (d ^ "/hip/hip_fp16.h"))
  in
  List.dedup_and_sort ~compare:String.compare (from_backend @ sdk)

let capabilities () = Hip_backend.host_capabilities ()
