(* gh-ocannl-1688 follow-up: every vendor-gated `select` arm is copied under arrayjit/vendor_stubs/
   and compiled there.

   WHY. A test or tool executable reaches CUDA, HIP or Metal code through a dune `select` whose
   vendor arm is chosen only where the vendor library is installed, and whose fallback stub is
   chosen everywhere else -- silently, so `dune build @check` off the vendor's box compiles the stub
   and says nothing about the arm. `arrayjit/vendor_stubs/` (gh-ocannl-1688) closes that by
   compiling a copy of each arm against interface-only stand-ins for the vendor packages, but the
   copies are hand-kept: a NEW arm that `arrayjit/vendor_stubs/arms/dune` does not copy is
   type-checked nowhere off its box, and nothing says so. This scan says so.

   WHAT IT READS. Every dune file in the repository. A `select` choice is VENDOR-GATED when a
   library its condition requires (not one it forbids with `!`) belongs to a package the stand-ins
   cover, or is one of the vendor backends the stand-ins compile. Both sets are DERIVED from the
   stand-in tree rather than listed here: a package is the `<package>` of a `(library (name
   <package>_stub))` in `arrayjit/vendor_stubs/<package>/dune`, and a vendor backend is the
   `<backend>` of a `(library (name <backend>_stub_check))` anywhere under `arrayjit/vendor_stubs/`,
   together with the public name the corpus gives the library `<backend>`. Either set coming up
   empty is refused, since it would make every arm look ungated. The fallback `(-> …)` is never
   demanded: it is what compiles off the box already.

   THE RULE. Each vendor-gated arm needs a COPY under `arrayjit/vendor_stubs/` whose source resolves
   to the arm's own file -- a rule action `(copy# <src> <dst>)` (or `copy`), or a `copy_files#` (or
   `copy_files`) glob, `{a,b}` alternatives and `*`/`?` wildcards expanded -- and the copied module
   has to be compiled by a `library` of the directory it lands in, by its `(modules …)` field or
   dune's default set. A copy compiling nowhere checks nothing, so it is refused as well. A copy
   elsewhere in the repository, or of a same-named file from another directory, counts for nothing.

   WHAT IT DOES NOT READ. Whether the copy keeps the module name the executable gives the arm, or
   compiles with the executable's libraries: those are `arms/dune`'s conventions, and a deviation
   still compiles the arm's text. A vendor package with no stand-in at all is outside its reach: no
   arm gated on it can be recognised, and adding the stand-in is what brings it in. A `[…]` set in a
   copy glob is read as matching nothing, which errs toward a demand. *)

open Base
open Stdio
open Verdict.Claims

(* Every file this scan reads arrives as a `@<path>` response file, because the list is longer than
   a Windows command line may be; see [Test_scan.Scan_argv]. *)
let argv = Test_scan.Scan_argv.expand Stdlib.Sys.argv

module Scan = Test_scan.Dune_stanza_scan

let stubs_root = "arrayjit/vendor_stubs"
let package_suffix = "_stub"
let backend_suffix = "_stub_check"

(* One stanza of one dune file, in the directory dune applies it to. *)
type stanza = { dune_file : string; dir : string; line : int; sexp : Sexp.t }

let site s = Printf.sprintf "%s:%d" s.dune_file s.line
let dir_of path = match Stdlib.Filename.dirname path with "." -> "" | dir -> dir

(* The stanzas of one dune file with their lines, [(subdir …)] resolved; a file the shared reader
   cannot read is refused rather than skipped. *)
let stanzas_of ~fail (dune_file, content) =
  match Scan.spelled_stanzas content with
  | exception exn ->
      fail
        (Printf.sprintf
           "%s does not read as a dune file (%s) -- its `select` arms and stand-in copies went \
            unchecked"
           dune_file (Stdlib.Printexc.to_string exn));
      []
  | pairs ->
      let rec walk dir pairs =
        List.concat_map pairs ~f:(fun (sexp, (raw : Scan.raw_form)) ->
            match (sexp, raw.raw_children) with
            | Sexp.List (Sexp.Atom "subdir" :: Sexp.Atom sub :: body), _ :: _ :: raw_body
              when List.length body = List.length raw_body ->
                walk (Scan.in_subdir dir sub) (List.zip_exn body raw_body)
            | _ -> [ { dune_file; dir; line = Scan.line_of content raw.raw_start; sexp } ])
      in
      walk (dir_of dune_file) pairs

let library_name s =
  match (Scan.head s.sexp, Scan.field s.sexp "name") with
  | Some "library", Some [ Sexp.Atom name ] -> Some name
  | _ -> None

let under_stubs dir = String.equal dir stubs_root || String.is_prefix dir ~prefix:(stubs_root ^ "/")

(* [{a,b}] alternatives, each group expanded in turn; no nesting. *)
let rec expand_braces pattern =
  match String.index pattern '{' with
  | None -> [ pattern ]
  | Some open_ -> (
      match String.index_from pattern open_ '}' with
      | None -> [ pattern ]
      | Some close ->
          let prefix = String.prefix pattern open_ in
          let suffix = String.drop_prefix pattern (close + 1) in
          String.sub pattern ~pos:(open_ + 1) ~len:(close - open_ - 1)
          |> String.split ~on:','
          |> List.concat_map ~f:(fun alternative -> expand_braces (prefix ^ alternative ^ suffix)))

(* A copy under the stand-in tree: which repository file it reads, and the module it makes. *)
type copy = {
  copy_stanza : stanza;
  reads : string -> bool;  (** whether it copies this repository-relative path *)
  module_of : string -> string;  (** the module the copy of that path compiles as *)
}

let module_of_file file = String.lowercase (Stdlib.Filename.remove_extension file)
let pform atom = String.is_substring atom ~substring:"%{"

let copies_of s =
  let resolve path = Scan.normalize_path (Scan.in_subdir s.dir path) in
  match Scan.head s.sexp with
  | Some "rule" ->
      let rec copies = function
        | Sexp.List [ Sexp.Atom ("copy" | "copy#"); Sexp.Atom src; Sexp.Atom dst ]
          when not (pform src || pform dst) ->
            let source = resolve src in
            [
              {
                copy_stanza = s;
                reads = String.equal source;
                module_of = (fun _ -> module_of_file (Stdlib.Filename.basename dst));
              };
            ]
        | Sexp.List l -> List.concat_map l ~f:copies
        | Sexp.Atom _ -> []
      in
      copies s.sexp
  | Some ("copy_files" | "copy_files#") ->
      let globs =
        match Scan.field s.sexp "files" with
        | Some args -> List.filter_map args ~f:(function Sexp.Atom g -> Some g | _ -> None)
        | None -> ( match s.sexp with Sexp.List (_ :: Sexp.Atom glob :: _) -> [ glob ] | _ -> [])
      in
      List.filter_map globs ~f:(fun glob ->
          if pform glob then None
          else
            let from_dir = resolve (dir_of glob) in
            let patterns = expand_braces (Stdlib.Filename.basename glob) in
            Some
              {
                copy_stanza = s;
                reads =
                  (fun path ->
                    String.equal (dir_of path) from_dir
                    && List.exists patterns ~f:(fun pattern ->
                        (not (String.mem pattern '['))
                        && Scan.glob_could_match pattern ~name:(Stdlib.Filename.basename path)));
                module_of = (fun path -> module_of_file (Stdlib.Filename.basename path));
              })
  | _ -> []

(* Whether a library of the copy's own directory compiles [module_]. *)
let compiled_by ~all copy module_ =
  List.exists all ~f:(fun o ->
      String.equal o.dune_file copy.copy_stanza.dune_file
      && String.equal o.dir copy.copy_stanza.dir
      && Option.is_some (library_name o)
      &&
      match Scan.explicit_modules o.sexp with
      | Scan.Named modules ->
          List.mem (List.map modules ~f:String.lowercase) module_ ~equal:String.equal
      | Scan.Default_less excluded ->
          not (List.mem (List.map excluded ~f:String.lowercase) module_ ~equal:String.equal))

(* What the census says of a whole tree. [vendor] pairs each vendor-gated arm's file with the copy
   compiling it, [None] where a refusal named it; [ungated] lists the non-fallback arms gated on no
   vendor library. *)
type census = {
  packages : string list;
  backend_libraries : string list;
  vendor : (string * string option) list;
  ungated : string list;
}

let analyze ~fail ~dune_files =
  if List.is_empty dune_files then (
    fail "no dune files among the arguments -- the rule's dune-file glob matches nothing";
    { packages = []; backend_libraries = []; vendor = []; ungated = [] })
  else
    let all = List.concat_map dune_files ~f:(stanzas_of ~fail) in
    let packages =
      List.filter_map all ~f:(fun s ->
          match library_name s with
          | Some name when String.equal (dir_of s.dir) stubs_root ->
              let package = Stdlib.Filename.basename s.dir in
              if String.equal name (package ^ package_suffix) then Some package else None
          | _ -> None)
      |> List.dedup_and_sort ~compare:String.compare
    in
    let backends =
      List.filter_map all ~f:(fun s ->
          match library_name s with
          | Some name when under_stubs s.dir -> String.chop_suffix name ~suffix:backend_suffix
          | _ -> None)
      |> List.dedup_and_sort ~compare:String.compare
    in
    if List.is_empty packages then
      fail
        "no library under arrayjit/vendor_stubs/<package>/dune is named `<package>_stub` -- the \
         vendor packages the stand-ins cover cannot be read, so no `select` arm gated on one can \
         be recognised";
    if List.is_empty backends then
      fail
        "no library under arrayjit/vendor_stubs/ is named `<backend>_stub_check` -- the vendor \
         backends the stand-ins compile cannot be read, so an arm gated on a backend library alone \
         would go undemanded";
    (* A backend by its private name and by the public one the corpus gives it. *)
    let backend_libraries =
      backends
      @ List.filter_map all ~f:(fun s ->
          match (library_name s, Scan.field s.sexp "public_name") with
          | Some name, Some [ Sexp.Atom public ] when List.mem backends name ~equal:String.equal ->
              Some public
          | _ -> None)
      |> List.dedup_and_sort ~compare:String.compare
    in
    let vendor_library library =
      List.mem packages (List.hd_exn (String.split library ~on:'.')) ~equal:String.equal
      || List.mem backend_libraries library ~equal:String.equal
    in
    let copies = List.concat_map all ~f:(fun s -> if under_stubs s.dir then copies_of s else []) in
    let vendor = ref [] and ungated = ref [] in
    List.iter all ~f:(fun s ->
        List.iter (Scan.select_choices s.sexp) ~f:(fun (target, condition, arm) ->
            let required =
              List.filter condition ~f:(fun l -> not (String.is_prefix l ~prefix:"!"))
            in
            let path = Scan.normalize_path (Scan.in_subdir s.dir arm) in
            if List.is_empty condition then ()
            else if not (List.exists required ~f:vendor_library) then ungated := path :: !ungated
            else
              let reading = List.filter copies ~f:(fun c -> c.reads path) in
              match List.find reading ~f:(fun c -> compiled_by ~all c (c.module_of path)) with
              | Some c ->
                  vendor :=
                    (path, Some (Printf.sprintf "%s/%s.ml" c.copy_stanza.dir (c.module_of path)))
                    :: !vendor
              | None -> (
                  vendor := (path, None) :: !vendor;
                  match reading with
                  | c :: _ ->
                      fail
                        (Printf.sprintf
                           "%s selects %s, and %s copies it into `%s.ml` of %s, where no library \
                            compiles that module -- the copy type-checks nothing; name `%s` in the \
                            `(modules …)` of a library there"
                           (site s) path (site c.copy_stanza) (c.module_of path) c.copy_stanza.dir
                           (c.module_of path))
                  | [] ->
                      fail
                        (Printf.sprintf
                           "%s selects `%s` for `%s` where %s is installed, and nothing under \
                            arrayjit/vendor_stubs/ copies %s -- off a box carrying that package \
                            the arm is type-checked nowhere, silently (gh-ocannl-1688). Copy it \
                            into arrayjit/vendor_stubs/arms/ as `%s` and compile it in a library \
                            there, as arms/dune does for its siblings"
                           (site s) arm target (String.concat ~sep:" " required) path target))));
    {
      packages;
      backend_libraries;
      vendor = List.dedup_and_sort !vendor ~compare:Poly.compare;
      ungated = List.dedup_and_sort !ungated ~compare:String.compare;
    }

(* Synthetic trees: the repository satisfies the rule, so its own corpus cannot tell a rule from a
   tautology. Each case builds the smallest tree holding one violation, or the nearest legitimate
   one, and claims on what [analyze] says of it. *)

let stand_ins =
  [
    ("arrayjit/vendor_stubs/cudajit/dune", "(library (name cudajit_stub))\n");
    ("arrayjit/vendor_stubs/cudajit/typecheck/dune", "(library (name cuda_backend_stub_check))\n");
    ( "arrayjit/lib/dune",
      "(library (name cuda_backend) (public_name arrayjit.cuda_backend) (optional))\n" );
  ]

let program ?(condition = "cudajit.cuda") ?(arm = "probe_impl.cudajit.ml") () =
  Printf.sprintf
    "(executable\n\
    \ (name probe)\n\
    \ (libraries base (select probe_impl.ml from (%s -> %s) (-> probe_impl.missing.ml))))\n"
    condition arm

let arms_copy ?(src = "../../../t/probe_impl.cudajit.ml") ?(modules = "probe_impl") () =
  ( "arrayjit/vendor_stubs/arms/dune",
    Printf.sprintf
      "(rule (action (copy# %s probe_impl.ml)))\n(library (name probe_arm) (modules %s))\n" src
      modules )

let control () =
  let source = "test/operations/vendor_stub_arms_scan.ml" in
  let run_tree dune_files =
    let messages = ref [] in
    let census = analyze ~fail:(fun m -> messages := m :: !messages) ~dune_files in
    (List.rev !messages, census)
  in
  let refused label ~format ~fragment dune_files =
    let messages, _ = run_tree dune_files in
    let routed =
      match messages with [ m ] -> String.is_substring m ~substring:fragment | _ -> false
    in
    if not routed then
      List.iter messages ~f:(fun m -> eprintf "  %s said: %s\n" label (Verdict.one_line m));
    if routed then Test_scan.Refusal_control_manifest.observe_failure ~source ~format;
    p label routed
  in
  (* The whole answer against the expected one: no refusal, and the census's two lists -- so a
     reader gone blind (no arm found) cannot pass an accepted case. *)
  let accepted label ~vendor ~ungated dune_files =
    let messages, census = run_tree dune_files in
    List.iter messages ~f:(fun m -> eprintf "  %s said: %s\n" label (Verdict.one_line m));
    p label (Poly.equal (messages, census.vendor, census.ungated) ([], vendor, ungated))
  in
  let uncopied_format =
    "%s selects `%s` for `%s` where %s is installed, and nothing under arrayjit/vendor_stubs/ \
     copies %s -- off a box carrying that package the arm is type-checked nowhere, silently \
     (gh-ocannl-1688). Copy it into arrayjit/vendor_stubs/arms/ as `%s` and compile it in a \
     library there, as arms/dune does for its siblings"
  in
  let copied = [ ("t/probe_impl.cudajit.ml", Some "arrayjit/vendor_stubs/arms/probe_impl.ml") ] in
  accepted
    "a vendor arm copied and compiled under the stand-ins is accepted, its fallback undemanded"
    ~vendor:copied ~ungated:[]
    (stand_ins @ [ ("t/dune", program ()); arms_copy () ]);
  accepted "a vendor arm copied by a copy_files glob into a default-module library is accepted"
    ~vendor:[ ("t/probe_cuda.ml", Some "arrayjit/vendor_stubs/arms/probe_cuda.ml") ]
    ~ungated:[]
    (stand_ins
    @ [
        ("t/dune", program ~arm:"probe_cuda.ml" ());
        ( "arrayjit/vendor_stubs/arms/dune",
          "(copy_files# (files ../../../t/probe_cuda.{ml,mli}))\n(library (name probe_arm))\n" );
      ]);
  accepted "an arm gated on no vendor library is not demanded" ~vendor:[]
    ~ungated:[ "t/probe_impl.unix.ml" ]
    (stand_ins @ [ ("t/dune", program ~condition:"unix" ~arm:"probe_impl.unix.ml" ()) ]);
  accepted "an arm that only forbids a vendor library is not demanded" ~vendor:[]
    ~ungated:[ "t/probe_impl.none.ml" ]
    (stand_ins @ [ ("t/dune", program ~condition:"!cudajit.cuda" ~arm:"probe_impl.none.ml" ()) ]);
  refused "a vendor arm nothing copies is refused, naming its dune file" ~format:uncopied_format
    ~fragment:"t/dune:1 selects `probe_impl.cudajit.ml` for `probe_impl.ml` where cudajit.cuda"
    (stand_ins @ [ ("t/dune", program ()) ]);
  refused "an arm gated on a vendor backend's public name alone is demanded" ~format:uncopied_format
    ~fragment:"where arrayjit.cuda_backend is installed"
    (stand_ins @ [ ("t/dune", program ~condition:"arrayjit.cuda_backend" ()) ]);
  refused "a copy of a same-named file from another directory does not count"
    ~format:uncopied_format ~fragment:"copies t/probe_impl.cudajit.ml"
    (stand_ins @ [ ("t/dune", program ()); arms_copy ~src:"../../../u/probe_impl.cudajit.ml" () ]);
  refused "a copy outside arrayjit/vendor_stubs does not count" ~format:uncopied_format
    ~fragment:"copies t/probe_impl.cudajit.ml"
    (stand_ins
    @ [
        ("t/dune", program ());
        ( "u/dune",
          "(rule (action (copy# ../t/probe_impl.cudajit.ml probe_impl.ml)))\n\
           (library (name probe_arm) (modules probe_impl))\n" );
      ]);
  refused "a copied arm no library compiles is refused"
    ~format:
      "%s selects %s, and %s copies it into `%s.ml` of %s, where no library compiles that module \
       -- the copy type-checks nothing; name `%s` in the `(modules …)` of a library there"
    ~fragment:"copies it into `probe_impl.ml` of arrayjit/vendor_stubs/arms"
    (stand_ins @ [ ("t/dune", program ()); arms_copy ~modules:"other" () ]);
  refused "a tree with no package stand-in is refused"
    ~format:
      "no library under arrayjit/vendor_stubs/<package>/dune is named `<package>_stub` -- the \
       vendor packages the stand-ins cover cannot be read, so no `select` arm gated on one can be \
       recognised"
    ~fragment:"named `<package>_stub`" (List.tl_exn stand_ins);
  refused "a tree with no backend stand-in is refused"
    ~format:
      "no library under arrayjit/vendor_stubs/ is named `<backend>_stub_check` -- the vendor \
       backends the stand-ins compile cannot be read, so an arm gated on a backend library alone \
       would go undemanded"
    ~fragment:"named `<backend>_stub_check`"
    [ List.hd_exn stand_ins ];
  refused "a dune file that does not read is refused"
    ~format:
      "%s does not read as a dune file (%s) -- its `select` arms and stand-in copies went unchecked"
    ~fragment:"t/dune does not read as a dune file"
    (stand_ins @ [ ("t/dune", "(executable (name probe)\n") ]);
  refused "an empty dune-file corpus is refused"
    ~format:"no dune files among the arguments -- the rule's dune-file glob matches nothing"
    ~fragment:"no dune files among the arguments" [];
  Test_scan.Refusal_control_manifest.print "vendor_stub_arms_scan.ml"

let () =
  if Array.length argv = 2 && String.equal argv.(1) "--control" then (
    control ();
    Stdlib.exit 0);
  if Array.length argv < 2 then (
    eprintf "Usage: %s <workspace_root> <dune files...>|--control\n" argv.(0);
    Stdlib.exit 1);
  let base = Scan.base_dir argv.(1) in
  let dune_files =
    Array.to_list (Array.subo argv ~pos:2)
    |> List.filter ~f:(fun path -> String.equal (Stdlib.Filename.basename path) "dune")
    |> List.map ~f:(fun path -> (Scan.repo_relative base path, path))
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
    |> List.map ~f:(fun (path, on_disk) -> (path, In_channel.read_all on_disk))
  in
  let census = analyze ~fail ~dune_files in
  (* The tally moves with every new vendor arm, so it stays off the golden; the floor is what says
     the reader is not blind. *)
  eprintf "Vendor-gated select arms: %d (not part of the golden).\n" (List.length census.vendor);
  p_all ~min:1 "every vendor-gated select arm the census finds is compiled by a stand-in copy"
    census.vendor ~f:(fun (_, copy) -> Option.is_some copy);
  printf "\nVendor packages with a stand-in: %s\n" (String.concat ~sep:", " census.packages);
  printf "Vendor backend libraries: %s\n" (String.concat ~sep:", " census.backend_libraries);
  printf "\nVendor-gated select arms, each with the stand-in copy that compiles it:\n";
  List.iter census.vendor ~f:(fun (arm, copy) ->
      printf "  %s -> %s\n" arm (Option.value copy ~default:"(no copy compiles it)"));
  printf "\nSelect arms gated on no vendor library (no copy demanded):\n";
  if List.is_empty census.ungated then printf "  (none)\n"
  else List.iter census.ungated ~f:(fun arm -> printf "  %s\n" arm);
  Verdict.success_trailer ~blank_line:true
    [ "every vendor-gated select arm is type-checked off its box, against the stand-ins." ];
  Test_scan.Refusal_control_manifest.print "vendor_stub_arms_scan.ml"
