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

let write_header dir ~mtime data =
  let path = file dir in
  Stdio.Out_channel.write_all path ~data;
  Unix.utimes path mtime mtime

let with_directory f =
  let dir = Unix.realpath (Stdlib.Filename.temp_dir "ocannl-host-headers-" "") in
  Exn.protect
    ~f:(fun () -> f dir)
    ~finally:(fun () ->
      if Stdlib.Sys.file_exists (file dir) then Unix.unlink (file dir);
      Unix.rmdir dir)

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
      Unix.utimes (file first) later later;
      let touched = record first in
      p "changing only header mtime moves the fingerprint" (not (String.equal initial touched));
      write_header first ~mtime:later "ab";
      let grown = record first in
      p "changing only header size moves the fingerprint" (not (String.equal touched grown));
      Unix.unlink (file first);
      p "removing the header identifies the missing file"
        (String.equal (record first) (first ^ "(missing:" ^ header ^ ")"));
      let leg =
        Verdict.leg ~aggregation:`Environment ~when_:(Unix.has_symlink ())
          ~on:"host without symlink support" ()
      in
      in_leg ~leg "repointing an include directory symlink moves its resolved path" (fun label ->
          write_header first ~mtime "a";
          with_directory (fun second ->
              write_header second ~mtime "a";
              let link = Stdlib.Filename.concat first "include" in
              let through_link target =
                Unix.symlink ~to_dir:true target link;
                Exn.protect ~f:(fun () -> record link) ~finally:(fun () -> Unix.unlink link)
              in
              let digest = String.chop_prefix_exn initial ~prefix:first in
              let linked_first = through_link first in
              let linked_second = through_link second in
              p label
                (String.equal linked_first (link ^ "=>" ^ first ^ digest)
                && String.equal linked_second (link ^ "=>" ^ second ^ digest)))))
