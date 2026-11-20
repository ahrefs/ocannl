open Base
open Verdict.Claims
module Cuda = Ir.Cuda_like_config.Cuda
module Hip = Ir.Cuda_like_config.Hip

let expect_headers label actual expected =
  let sorted = List.sort ~compare:String.compare in
  let matches = List.equal String.equal (sorted actual) (sorted expected) in
  if not matches then
    Stdio.eprintf "%s: got [%s], expected [%s]\n" label (String.concat ~sep:", " actual)
      (String.concat ~sep:", " expected);
  p label matches

let header = "toolkit.h"
let file dir = Stdlib.Filename.concat dir header
let record dir = Utils.capability_dir ~requires:[ header ] (Some dir)
let atime = 1_700_000_000.

let write_header dir ~mtime data =
  let path = file dir in
  Stdio.Out_channel.write_all path ~data;
  Unix.utimes path atime mtime

let try_directory_symlink ~target link =
  match Unix.symlink ~to_dir:true target link with
  | () -> true
  | exception Unix.Unix_error ((EACCES | EPERM | ENOSYS | EOPNOTSUPP), _, _) -> false
  | exception Unix.Unix_error _ when Stdlib.Sys.win32 && not (Unix.has_symlink ()) -> false

let with_directory f =
  let dir = Unix.realpath (Stdlib.Filename.temp_dir "ocannl-host-headers-" "") in
  Exn.protect
    ~f:(fun () -> f dir)
    ~finally:(fun () ->
      if Stdlib.Sys.file_exists (file dir) then Unix.unlink (file dir);
      Unix.rmdir dir)

let check_repointed_symlink first ~mtime ~initial =
  let label = "repointing an include directory symlink moves its resolved path" in
  let link = Stdlib.Filename.concat first "include" in
  let linked_record () =
    Exn.protect ~f:(fun () -> record link) ~finally:(fun () -> Unix.unlink link)
  in
  write_header first ~mtime "a";
  if try_directory_symlink ~target:first link then
    let linked_first = linked_record () in
    with_directory (fun second ->
        write_header second ~mtime "a";
        Unix.symlink ~to_dir:true second link;
        let linked_second = linked_record () in
        let digest = String.chop_prefix_exn initial ~prefix:first in
        p label
          (String.equal linked_first (link ^ "=>" ^ first ^ digest)
          && String.equal linked_second (link ^ "=>" ^ second ^ digest)))
  else
    skipped ~aggregation:`Environment ~backend:"temporary directory without symlink support" label

let check_macos_build_version read =
  let system_version_plist = Stdlib.Filename.temp_file "ocannl-system-version-" ".plist" in
  let probe build =
    Stdio.Out_channel.write_all system_version_plist
      ~data:
        (Printf.sprintf
           {|<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
  <key>ProductVersion</key><string>15.1</string>
  <key>ProductBuildVersion</key><string>%s</string>
</dict>
</plist>
|}
           build);
    read ~system_version_plist
  in
  Exn.protect
    ~f:(fun () ->
      let first = probe "24B83" in
      let second = probe "24B91" in
      let find = List.Assoc.find ~equal:String.equal in
      let api = find first "metal_math_api" in
      Option.is_some api
      && Option.equal String.equal api (find second "metal_math_api")
      && Option.equal String.equal (find first "metal_driver_build") (Some "24B83")
      && Option.equal String.equal (find second "metal_driver_build") (Some "24B91"))
    ~finally:(fun () -> Unix.unlink system_version_plist)

let () =
  (* Independent expectations make a directive the parser silently drops observable. *)
  expect_headers "CUDA prelude names its toolkit headers" (Cuda.prelude_headers ())
    [ "cuda_fp16.h"; "cuda_bf16.h"; "cuda_fp8.h" ];
  expect_headers "CUDA conditional includes name their toolkit headers"
    (Cuda.conditional_headers ()) [ "mma.h" ];
  expect_headers "HIP prelude names its toolkit headers" (Hip.prelude_headers ())
    [ "hip/hip_fp16.h"; "hip/hip_bf16.h"; "hip/hip_fp8.h" ];
  expect_headers "HIP conditional includes name their toolkit headers" (Hip.conditional_headers ())
    [ "rocwmma/rocwmma.hpp" ];
  with_directory (fun first ->
      let mtime = 1_700_000_000. in
      write_header first ~mtime "a";
      let initial = record first in
      let later = mtime +. 10. in
      Unix.utimes (file first) atime later;
      let touched = record first in
      p "changing only header mtime moves the fingerprint" (not (String.equal initial touched));
      write_header first ~mtime:later "ab";
      let grown = record first in
      p "changing only header size moves the fingerprint" (not (String.equal touched grown));
      Unix.unlink (file first);
      p "removing the header identifies the missing file"
        (String.equal (record first) (first ^ "(missing:" ^ header ^ ")"));
      check_repointed_symlink first ~mtime ~initial);
  let label = "changing the macOS build moves the host record with the same Metal math API" in
  match Host_capability_fingerprint_metal.read with
  | None -> skipped ~aggregation:`Environment ~backend:"host without Metal" label
  | Some read -> p label (check_macos_build_version read)
