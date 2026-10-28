(* The Metal arm of tools/vendor_idents.ml (gh-ocannl-1338): the backend's own probe, the Metal
   toolchain's header tree where one is installed, and the backend's host capabilities. The runtime
   compiler ([newLibraryWithSource:]) carries its headers inside the framework, so on a box without
   the Metal toolchain no header is found and only the table's own entries are measured. *)

open Base

let built = true
let probe () = Metal_backend.Impl.routine_name_probe ()

(* The [include/metal] parent of a Metal toolchain's clang: Xcode's, or the downloadable toolchain
   component's. [metal_stdlib] is looked up under it. *)
let include_dirs () =
  let candidates =
    [
      "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/metal/macos/lib/clang";
      "/Library/Developer/Toolchains/Metal.xctoolchain/usr/metal/current/lib/clang";
    ]
  in
  List.concat_map candidates ~f:(fun root ->
      if Stdlib.Sys.file_exists root && Stdlib.Sys.is_directory root then
        Stdlib.Sys.readdir root |> Array.to_list
        |> List.map ~f:(fun v -> root ^ "/" ^ v ^ "/include/metal")
        |> List.filter ~f:(fun d -> Stdlib.Sys.file_exists (d ^ "/metal_stdlib"))
      else [])

let capabilities () = Metal_backend.host_capabilities ()
