open Base
open Verdict.Claims

let ( / ) = Stdlib.Filename.concat
let write path data = Stdio.Out_channel.write_all path ~data

let rec remove_tree path =
  match Unix.lstat path with
  | { Unix.st_kind = Unix.S_DIR; _ } ->
      Array.iter (Stdlib.Sys.readdir path) ~f:(fun child -> remove_tree (path / child));
      Unix.rmdir path
  | _ -> Unix.unlink path

let try_file_symlink ~target link =
  match Unix.symlink ~to_dir:false target link with
  | () -> true
  | exception Unix.Unix_error ((EACCES | EPERM | ENOSYS | EOPNOTSUPP), _, _) -> false
  | exception Unix.Unix_error _ when Stdlib.Sys.win32 && not (Unix.has_symlink ()) -> false

let check_fixture root =
  let sdk = root / "sdk" in
  let hip = sdk / "hip" in
  let left = hip / "left" and right = hip / "right" in
  List.iter [ sdk; hip; left; right ] ~f:(fun path -> Unix.mkdir path 0o700);
  let shared = hip / "shared.h" and outside = root / "outside.h" in
  let outside_entry = "hip/outside.h" in
  write shared "int shared_header;\n#include \"sibling.h\"\n";
  write (left / "sibling.h") "int left_sibling;\n";
  write (right / "sibling.h") "int right_sibling;\n";
  (* A lookup from the link target would find this header instead of either sibling. *)
  write (hip / "sibling.h") "int target_sibling;\n";
  write outside "int outside_sdk;\n";
  (* The temporary filesystem can refuse links even when the host supports them. *)
  let can_link = try_file_symlink ~target:shared (left / "alias.h") in
  let leg =
    Verdict.leg ~aggregation:`Environment ~when_:can_link
      ~on:"temporary directory without file symlink support" ()
  in
  let harvest =
    lazy
      (Unix.symlink ~to_dir:false shared (right / "alias.h");
       Unix.symlink ~to_dir:false outside (sdk / outside_entry);
       let dirs = [ sdk ] in
       let entries =
         [ (`Angled, "hip/left/alias.h"); (`Angled, "hip/right/alias.h"); (`Angled, outside_entry) ]
       in
       Vendor_idents_headers.header_closure ~dirs ~entries
         ~header_identity:(Vendor_idents_headers.hip_sdk_header_identity dirs))
  in
  Verdict.in_leg ~leg "the shared HIP header and two siblings account for three reads" (fun label ->
      let reads, _, _ = Lazy.force harvest in
      p label (reads = 3));
  Verdict.in_leg ~leg "both HIP header aliases resolve their own quoted siblings" (fun label ->
      let _, _, tokens = Lazy.force harvest in
      p label
        (Hash_set.mem tokens "shared_header"
        && Hash_set.mem tokens "left_sibling"
        && Hash_set.mem tokens "right_sibling"
        && not (Hash_set.mem tokens "target_sibling")));
  Verdict.in_leg ~leg "the HIP SDK boundary rejects the external symlink" (fun label ->
      let _, unresolved, tokens = Lazy.force harvest in
      p label
        (List.equal String.equal unresolved [ outside_entry ]
        && not (Hash_set.mem tokens "outside_sdk")))

let () =
  let root = Stdlib.Filename.temp_dir "hip-sdk-header-symlinks" "" in
  Exn.protect
    ~f:(fun () -> check_fixture (Unix.realpath root))
    ~finally:(fun () -> remove_tree root)
