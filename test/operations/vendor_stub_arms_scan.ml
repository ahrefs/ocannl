(* gh-ocannl-1688 follow-up: every vendor-gated `select` arm is copied under arrayjit/vendor_stubs/
   and compiled there; gh-ocannl-2288: with nothing its genuine stanza lacks.

   WHY. A test or tool executable reaches CUDA, HIP or Metal code through a dune `select` whose
   vendor arm is chosen only where the vendor library is installed, and whose fallback stub is
   chosen everywhere else -- silently, so `dune build @check` off the vendor's box compiles the stub
   and says nothing about the arm. `arrayjit/vendor_stubs/` (gh-ocannl-1688) closes that by
   compiling a copy of each arm against interface-only stand-ins for the vendor packages, but the
   copies are hand-kept: a NEW arm that `arrayjit/vendor_stubs/arms/dune` does not copy is
   type-checked nowhere off its box, and nothing says so. This scan says so. And their stanzas are
   hand-kept too: a stand-in compiling with a library its genuine stanza does not declare is a green
   `@check` off the box and a red build on it (gh-ocannl-2234: the genuine CUDA backend lacked
   `pprint`, its stand-in did not). This scan says that as well.

   WHAT IT READS. Every dune file in the repository. A `select` choice is VENDOR-GATED when a
   library its condition requires (not one it forbids with `!`) belongs to a package the stand-ins
   cover, or is one of the vendor backends the stand-ins compile. Both sets are DERIVED from the
   stand-in tree rather than listed here: a package is the `<package>` of a `(library (name
   <package>_stub))` in `arrayjit/vendor_stubs/<package>/dune`, and a vendor backend -- any
   vendor-gated arrayjit library, `metal_math_api_runtime` among them -- is the `<backend>` of a
   `(library (name <backend>_stub_check))` anywhere under `arrayjit/vendor_stubs/`, together with
   the public name the corpus gives the library `<backend>`. Either set coming up empty is refused,
   since it would make every arm look ungated. The fallback `(-> …)` is never demanded: it is what
   compiles off the box already.

   THE RULE. Each vendor-gated arm needs a COPY under `arrayjit/vendor_stubs/` whose source resolves
   to the arm's own file -- a rule action `(copy# <src> <dst>)` (or `copy`), or a `copy_files#` (or
   `copy_files`) glob, `{a,b}` alternatives and `*`/`?` wildcards expanded -- and the copied module
   has to be compiled by a `library` of the directory it lands in, by its `(modules …)` field or
   dune's default set. A copy compiling nowhere checks nothing, so it is refused as well. A copy
   elsewhere in the repository, or of a same-named file from another directory, counts for nothing.

   THE RULE, LIBRARIES. Every library of the stand-in tree but a package stand-in is held to the
   GENUINE stanza of each module it compiles from a copy: the stanza whose `select` chooses the
   copied arm, or else the stanza of the source's directory whose `(modules …)` names it, or else
   the one whose default set takes it -- none, or several, is refused. Each library the stand-in
   declares is mapped through the substitution table, derived from the two sets above rather than
   written down: `<package>_stub` stands for any library of `<package>` (`cudajit.cuda`,
   `cudajit.nvrtc`), `<backend>_stub_check` for the library `<backend>` under either of its names,
   and any other library for itself, a corpus library's public and private names being one. A
   library of the stand-in tree outside the table stands for nothing, and is refused. A mapped
   library the genuine stanza does not declare -- in its `(libraries …)`, a `(re_export …)`, or a
   `select` condition, which links it wherever the arm is chosen -- is refused unless
   [availability_only] names it with its reason, and an entry there excusing nothing is refused in
   turn. A stand-in's `(re_export …)` needs the genuine stanza's, since it widens what a dependent
   sees. A preprocessor the stand-in runs over the module and the genuine stanza does not is refused
   too. Only that direction is a defect: a genuine stanza declaring MORE is normal, since an
   executable carries modules its arm's library does not.

   WHAT IT DOES NOT READ. Whether the copy keeps the module name the executable gives the arm, an
   `arms/dune` convention: a deviation still compiles the arm's text. Which module of a package a
   copy reaches: `cudajit_stub` is one library for all of `cudajit`, so a genuine stanza naming
   `cudajit.cuda` alone satisfies it even where the copy uses `Nvrtc`. Whether the copy uses what
   its library declares: declarations are compared, never uses, so a stand-in library declared and
   unused is refused all the same where the genuine stanza lacks it -- drop it. Which arm of ANOTHER
   `select` a box chooses: a module is held to every condition of the stanza's other selects, as a
   stand-in compiling several arms declares their union (an arm's own select counts only through the
   arm's condition). A module that copies no file is held to nothing, and so is a default-set
   library's module no rule copies -- beside a `copy_files` glob, whose files dune files do not
   list, a default-set library is refused; the census lists the libraries compiling no copy. Of
   preprocessing, only the `pps` libraries: not their flags or arguments, not `preprocessor_deps`,
   and any other form (`staged_pps`, `action`, `future_syntax`) is refused as unread. A vendor
   package with no stand-in at all is outside its reach: no arm gated on it can be recognised, and
   adding the stand-in is what brings it in. A `[…]` set in a copy glob is read as matching nothing,
   which errs toward a demand. *)

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
  source_of : string -> string option;
      (** the repository file a module of that name is copied from, if this copy makes it *)
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
            let made = module_of_file (Stdlib.Filename.basename dst) in
            [
              {
                copy_stanza = s;
                reads = String.equal source;
                module_of = (fun _ -> made);
                source_of =
                  (fun m ->
                    if String.equal m made && String.is_suffix dst ~suffix:".ml" then Some source
                    else None);
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
            let reads path =
              String.equal (dir_of path) from_dir
              && List.exists patterns ~f:(fun pattern ->
                  (not (String.mem pattern '['))
                  && Scan.glob_could_match pattern ~name:(Stdlib.Filename.basename path))
            in
            Some
              {
                copy_stanza = s;
                reads;
                module_of = (fun path -> module_of_file (Stdlib.Filename.basename path));
                (* A module's implementation is what a glob copies of it; an interface-only copy
                   makes no module whose libraries matter. *)
                source_of =
                  (fun m ->
                    let path = Scan.in_subdir from_dir (m ^ ".ml") in
                    if reads path then Some path else None);
              })
  | _ -> []

(* How a stanza's `(modules …)` field claims the module [m]: by name, through dune's default set, or
   not at all. Choosing among a directory's stanzas, a [`Named] claim wins: dune's default set is
   what no explicit list claims. *)
let stanza_claims s m =
  match Scan.explicit_modules s.sexp with
  | Scan.Named modules ->
      if List.mem (List.map modules ~f:String.lowercase) m ~equal:String.equal then `Named else `No
  | Scan.Default_less excluded ->
      if List.mem (List.map excluded ~f:String.lowercase) m ~equal:String.equal then `No
      else `Default

(* Whether a library of the copy's own directory compiles [module_]. *)
let compiled_by ~all copy module_ =
  List.exists all ~f:(fun o ->
      String.equal o.dune_file copy.copy_stanza.dune_file
      && String.equal o.dir copy.copy_stanza.dir
      && Option.is_some (library_name o)
      && not (Poly.equal (stanza_claims o module_) `No))

(* The libraries a stanza declares, each with whether it re-exports it: its `(libraries …)` atoms, a
   `(re_export …)`'s, and the libraries a `select` condition requires, since the condition is what
   makes dune link them wherever the arm is chosen. For a module copied from [arm], the alternatives
   of that arm's own `select` count only through the arm's condition: dune links one alternative per
   select, so a sibling's libraries are absent wherever this arm is chosen. Other selects count with
   all their conditions: which of their arms a box chooses is not this arm's to say, and a stand-in
   library compiling several arms declares their union. *)
let declared_libraries ?arm s =
  let fields = Option.value (Scan.field s.sexp "libraries") ~default:[] in
  let own_select =
    match arm with
    | None -> None
    | Some arm ->
        List.find_map (Scan.select_choices s.sexp) ~f:(fun (target, _, a) ->
            if String.equal (Scan.normalize_path (Scan.in_subdir s.dir a)) arm then Some target
            else None)
  in
  List.concat_map fields ~f:(function
    | Sexp.Atom l -> [ (l, false) ]
    | Sexp.List [ Sexp.Atom "re_export"; Sexp.Atom l ] -> [ (l, true) ]
    | _ -> [])
  @ List.concat_map (Scan.select_choices s.sexp) ~f:(fun (target, condition, a) ->
      let counts =
        match (own_select, arm) with
        | Some own, Some arm when String.equal own target ->
            String.equal (Scan.normalize_path (Scan.in_subdir s.dir a)) arm
        | _ -> true
      in
      if not counts then []
      else
        List.filter_map condition ~f:(fun l ->
            if String.is_prefix l ~prefix:"!" then None else Some (l, false)))

let stanza_label s =
  Printf.sprintf "(%s %s) of %s"
    (Option.value (Scan.head s.sexp) ~default:"?")
    (String.concat ~sep:" " (Scan.names_of s.sexp))
    s.dune_file

(* What a library a stand-in declares stands for in the genuine tree: a package stand-in for every
   library of its package, a backend's `_stub_check` copy for the backend library, and anything else
   for itself. *)
type substitute = Package of string | Library of string

let substitute_text = function
  | Package p -> Printf.sprintf "a library of package `%s`" p
  | Library l -> Printf.sprintf "`%s`" l

(* An AVAILABILITY-ONLY exception: a library a stand-in declares only because it is a stand-in, so
   that its genuine stanza rightly goes without it. Each names the stand-in library, the library,
   and why; one nothing needs any more is refused, so the list cannot outlive its reasons. *)
type exception_ = { stand_in : string; library : string; reason : string }

let availability_only : exception_ list = []

(* What the census says of a whole tree. [vendor] pairs each vendor-gated arm's file with the copy
   compiling it, [None] where a refusal named it; [ungated] lists the non-fallback arms gated on no
   vendor library. *)
type census = {
  packages : string list;
  backend_libraries : string list;
  vendor : (string * string option) list;
  ungated : string list;
  mirrors : (string * (string * string) list) list;
      (** each stand-in library with the copied modules it compiles, each with the genuine stanza
          its libraries and preprocessors are held to *)
  uncopied : string list;  (** the stand-in tree's other libraries, compiling no copy *)
  paired : (string * string) list;  (** each paired copy's source with the copy's own path *)
}

(* The preprocessors dune runs over module [m] of a stanza: a `(pps …)`'s libraries before any `--`,
   less its flags; under `per_module`, the spec of the group naming [m], and none where no group
   names it; none under `no_preprocessing` or an absent field. Any other form -- `staged_pps`, an
   `action`, `future_syntax` -- is not read: it comes back as its text, for a refusal. *)
let preprocessors s m =
  let of_spec = function
    | Sexp.Atom "no_preprocessing" -> Ok []
    | Sexp.List (Sexp.Atom "pps" :: args) ->
        Ok
          (List.take_while args ~f:(fun a -> not (Sexp.equal a (Sexp.Atom "--")))
          |> List.filter_map ~f:(function
            | Sexp.Atom a when not (String.is_prefix a ~prefix:"-" || String.is_prefix a ~prefix:"+")
              ->
                Some a
            | _ -> None))
    | spec -> Error (Sexp.to_string spec)
  in
  let names_m = function
    | Sexp.Atom n -> String.equal (String.lowercase n) m
    | Sexp.List _ -> false
  in
  match Scan.field s.sexp "preprocess" with
  | None -> Ok []
  | Some [ Sexp.List (Sexp.Atom "per_module" :: groups) ] as field -> (
      if
        not
          (List.for_all groups ~f:(function
            | Sexp.List (_ :: _ :: modules) ->
                List.for_all modules ~f:(function Sexp.Atom _ -> true | Sexp.List _ -> false)
            | _ -> false))
      then Error (Sexp.to_string (Sexp.List (Option.value_exn field)))
      else
        match
          List.find_map groups ~f:(function
            | Sexp.List (spec :: modules) when List.exists modules ~f:names_m -> Some spec
            | _ -> None)
        with
        | Some spec -> of_spec spec
        | None -> Ok [])
  | Some [ spec ] -> of_spec spec
  | Some other -> Error (Sexp.to_string (Sexp.List other))

(* One module a stand-in library compiles from a copy, with the genuine stanza compiling the copy's
   source and the module that source is there: an arm compiles as its `select` target. *)
type pair = {
  module_ : string;
  source : string;
  genuine : stanza;
  genuine_module : string;
  arm : bool;  (** whether [source] is an arm of a `select` of [genuine] *)
}

(* The library half of the rule: every library of the stand-in tree other than the package
   stand-ins, held module by copied module to the genuine stanza that compiles the copy's source.
   Returns the libraries with their pairs, and the libraries compiling no copy. *)
let mirrors ~fail ~exceptions ~all ~packages ~backends ~copies =
  (* One refusal per distinct message: several modules of one library share a genuine stanza. *)
  let refused = Hash_set.create (module String) in
  let fail message = if Hash_set.strict_add refused message |> Result.is_ok then fail message in
  let library_stanzas = List.filter all ~f:(fun s -> Option.is_some (library_name s)) in
  (* A library's two names, private and public, are one library. *)
  let canonical l =
    Option.value ~default:l
      (List.find_map library_stanzas ~f:(fun s ->
           match Scan.field s.sexp "public_name" with
           | Some [ Sexp.Atom public ] when String.equal public l -> library_name s
           | _ -> None))
  in
  let package_stand_in name =
    List.find packages ~f:(fun p -> String.equal name (p ^ package_suffix))
  in
  let stand_in_names =
    List.filter_map library_stanzas ~f:(fun s -> if under_stubs s.dir then library_name s else None)
  in
  (* The substitution table, read off the same derived sets the arm census uses. *)
  let substitute name =
    match package_stand_in name with
    | Some p -> Some (Package p)
    | None -> (
        match String.chop_suffix name ~suffix:backend_suffix with
        | Some b when List.mem backends b ~equal:String.equal -> Some (Library b)
        | _ ->
            if List.mem stand_in_names name ~equal:String.equal then None
            else Some (Library (canonical name)))
  in
  (* A re-export widens what a dependent sees, so a stand-in's re-export needs the genuine
     stanza's. *)
  let satisfies ~re_export declared sub =
    List.exists declared ~f:(fun (d, d_re_export) ->
        ((not re_export) || d_re_export)
        &&
        match sub with
        | Package p -> String.equal d p || String.is_prefix d ~prefix:(p ^ ".")
        | Library l -> String.equal (canonical d) l)
  in
  (* The stanza choosing each vendor-or-not arm, by the arm's path, with the module it makes. *)
  let arm_owners =
    List.concat_map all ~f:(fun s ->
        List.filter_map (Scan.select_choices s.sexp) ~f:(fun (target, condition, arm) ->
            if List.is_empty condition then None
            else Some (Scan.normalize_path (Scan.in_subdir s.dir arm), (s, module_of_file target))))
  in
  let genuine_of source =
    match List.Assoc.find arm_owners source ~equal:String.equal with
    | Some (owner, m) -> [ (owner, m, true) ]
    | None -> (
        let m = module_of_file (Stdlib.Filename.basename source) in
        let bearing =
          List.filter all ~f:(fun o ->
              String.equal o.dir (dir_of source)
              && (not (under_stubs o.dir))
              && List.mem Scan.module_bearing_heads
                   (Option.value (Scan.head o.sexp) ~default:"")
                   ~equal:String.equal)
        in
        (* Dune's default set is what no explicit list claims. *)
        let claiming how = List.filter bearing ~f:(fun o -> Poly.equal (stanza_claims o m) how) in
        match claiming `Named with
        | _ :: _ as named -> List.map named ~f:(fun o -> (o, m, false))
        | [] -> List.map (claiming `Default) ~f:(fun o -> (o, m, false)))
  in
  let used = Hash_set.create (module String) in
  let exception_key stand_in library = stand_in ^ " " ^ library in
  let checked =
    List.filter_map library_stanzas ~f:(fun l ->
        match library_name l with
        | Some name when under_stubs l.dir && Option.is_none (package_stand_in name) ->
            let own =
              List.filter copies ~f:(fun c ->
                  String.equal c.copy_stanza.dune_file l.dune_file
                  && String.equal c.copy_stanza.dir l.dir)
            in
            let modules =
              match Scan.explicit_modules l.sexp with
              | Scan.Named ms -> List.map ms ~f:String.lowercase
              | Scan.Default_less _
                when List.exists own ~f:(fun c ->
                         not
                           (Option.equal String.equal (Scan.head c.copy_stanza.sexp) (Some "rule")))
                ->
                  (* A glob copies files dune files do not list, so which modules the default set
                     then holds -- and which genuine stanzas they answer to -- is not readable. *)
                  fail
                    (Printf.sprintf
                       "%s compiles dune's default module set while a `copy_files` glob copies \
                        into %s -- which modules it compiles cannot be read from dune files, so \
                        their genuine stanzas cannot be found; list them in `(modules …)`"
                       (site l) l.dir);
                  []
              | Scan.Default_less excluded ->
                  (* What dune files alone can name of the default set: the modules a rule copies
                     here. *)
                  List.concat_map own ~f:(fun c ->
                      match Scan.head c.copy_stanza.sexp with
                      | Some "rule" -> [ c.module_of "" ]
                      | _ -> [])
                  |> List.filter ~f:(fun m ->
                      not (List.mem (List.map excluded ~f:String.lowercase) m ~equal:String.equal))
                  |> List.dedup_and_sort ~compare:String.compare
            in
            let pairs =
              List.concat_map modules ~f:(fun m ->
                  List.filter_map own ~f:(fun c -> c.source_of m)
                  |> List.dedup_and_sort ~compare:String.compare
                  |> List.filter_map ~f:(fun source ->
                      match genuine_of source with
                      | [ (genuine, genuine_module, arm) ] ->
                          Some { module_ = m; source; genuine; genuine_module; arm }
                      | found ->
                          fail
                            (Printf.sprintf
                               "%s compiles `%s`, a copy of %s, and %s stanzas of %s compile that \
                                file -- the stand-in's libraries cannot be held to a genuine \
                                stanza's; copy a file exactly one stanza compiles"
                               (site l) m source
                               (if List.is_empty found then "no" else "several")
                               (dir_of source));
                          None))
            in
            List.iter (declared_libraries l) ~f:(fun (library, re_export) ->
                match substitute library with
                | None ->
                    fail
                      (Printf.sprintf
                         "%s declares `%s`, another library of the stand-in tree, which stands for \
                          no genuine library -- the declaration cannot be held to a genuine \
                          stanza's; depend on the `<package>_stub` or `<backend>_stub_check` \
                          library it reaches instead"
                         (site l) library)
                | Some sub ->
                    let lacking =
                      List.filter pairs ~f:(fun p ->
                          not
                            (satisfies ~re_export
                               (declared_libraries
                                  ?arm:(if p.arm then Some p.source else None)
                                  p.genuine)
                               sub))
                    in
                    if not (List.is_empty lacking) then
                      if
                        List.exists exceptions ~f:(fun e ->
                            String.equal e.stand_in name && String.equal e.library library)
                      then Hash_set.add used (exception_key name library)
                      else
                        List.iter lacking ~f:(fun p ->
                            fail
                              (Printf.sprintf
                                 "%s compiles a copy of %s with `%s`, standing for %s, which %s, \
                                  the stanza compiling that file, does not declare -- the stand-in \
                                  type-checks where the genuine stanza would not (gh-ocannl-2234); \
                                  declare it there, or drop it from the stand-in"
                                 (site l) p.source
                                 (if re_export then "(re_export " ^ library ^ ")" else library)
                                 (substitute_text sub) (site p.genuine))));
            List.iter pairs ~f:(fun p ->
                match (preprocessors l p.module_, preprocessors p.genuine p.genuine_module) with
                | Error form, _ | _, Error form ->
                    fail
                      (Printf.sprintf
                         "%s compiles a copy of %s, and the `(preprocess …)` of %s or %s is not a \
                          form this scan reads (%s) -- its preprocessors cannot be compared; spell \
                          it as `(pps …)`, `per_module` over `(pps …)`, or `no_preprocessing`"
                         (site l) p.source (site l) (site p.genuine) form)
                | Ok stand_in, Ok genuine ->
                    List.iter stand_in ~f:(fun ppx ->
                        if
                          not
                            (List.mem (List.map genuine ~f:canonical) (canonical ppx)
                               ~equal:String.equal)
                        then
                          fail
                            (Printf.sprintf
                               "%s preprocesses `%s`, a copy of %s, with `%s`, which %s, the \
                                stanza compiling that file, does not run over it -- the stand-in \
                                type-checks where the genuine stanza would not; add it there, or \
                                drop it from the stand-in"
                               (site l) p.module_ p.source ppx (site p.genuine))));
            Some (name, l.dir, pairs)
        | _ -> None)
  in
  List.iter exceptions ~f:(fun e ->
      if not (Hash_set.mem used (exception_key e.stand_in e.library)) then
        fail
          (Printf.sprintf
             "the availability-only exception for `%s` in stand-in library `%s` (%s) excuses \
              nothing: the library declares no `%s` a genuine stanza it is held to lacks -- delete \
              the exception"
             e.library e.stand_in e.reason e.library));
  let checked = List.sort checked ~compare:(fun (a, _, _) (b, _, _) -> String.compare a b) in
  ( List.filter_map checked ~f:(fun (name, _, pairs) ->
        if List.is_empty pairs then None
        else Some (name, List.map pairs ~f:(fun p -> (p.module_, stanza_label p.genuine)))),
    List.filter_map checked ~f:(fun (name, _, pairs) ->
        if List.is_empty pairs then Some name else None),
    List.concat_map checked ~f:(fun (_, dir, pairs) ->
        List.map pairs ~f:(fun p -> (p.source, Printf.sprintf "%s/%s.ml" dir p.module_))) )

let analyze ?(exceptions = availability_only) ~fail ~dune_files () =
  if List.is_empty dune_files then (
    fail "no dune files among the arguments -- the rule's dune-file glob matches nothing";
    {
      packages = [];
      backend_libraries = [];
      vendor = [];
      ungated = [];
      mirrors = [];
      uncopied = [];
      paired = [];
    })
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
    let mirrors, uncopied, paired = mirrors ~fail ~exceptions ~all ~packages ~backends ~copies in
    {
      packages;
      backend_libraries;
      vendor = List.dedup_and_sort !vendor ~compare:Poly.compare;
      ungated = List.dedup_and_sort !ungated ~compare:String.compare;
      mirrors;
      uncopied;
      paired;
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

let arms_copy ?(src = "../../../t/probe_impl.cudajit.ml") ?(modules = "probe_impl")
    ?(libraries = "") () =
  ( "arrayjit/vendor_stubs/arms/dune",
    Printf.sprintf
      "(rule (action (copy# %s probe_impl.ml)))\n\
       (library (name probe_arm) (modules %s) (libraries %s))\n\
       (library (name probe_helper) (modules))\n"
      src modules libraries )

(* The library half's fixture: the genuine CUDA backend library on line 2 of its dune file, and its
   `typecheck/` stand-in on line 2 of its own. *)
let genuine_backend ?(preprocess = "") libraries =
  ( "arrayjit/lib/dune",
    Printf.sprintf
      "(library (name ir) (public_name arrayjit.ir))\n\
       (library (name cuda_backend) (public_name arrayjit.cuda_backend) (optional)\n\
      \ (modules cuda_backend) (libraries %s)%s)\n"
      libraries preprocess )

let backend_stand_in ?(preprocess = "") libraries =
  [
    ("arrayjit/vendor_stubs/cudajit/dune", "(library (name cudajit_stub))\n");
    ( "arrayjit/vendor_stubs/cudajit/typecheck/dune",
      Printf.sprintf
        "(copy_files# (files ../../../lib/cuda_backend.{ml,mli}))\n\
         (library (name cuda_backend_stub_check) (modules cuda_backend) (libraries %s)%s)\n"
        libraries preprocess );
  ]

let control () =
  let source = "test/operations/vendor_stub_arms_scan.ml" in
  let run_tree ?exceptions dune_files =
    let messages = ref [] in
    let census = analyze ?exceptions ~fail:(fun m -> messages := m :: !messages) ~dune_files () in
    (List.rev !messages, census)
  in
  let refused ?exceptions label ~format ~fragment dune_files =
    let messages, _ = run_tree ?exceptions dune_files in
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
  (* The library half's answer: no refusal, and every stand-in library paired as expected -- so a
     reader that pairs nothing cannot pass. *)
  let mirrored ?exceptions label ~mirrors dune_files =
    let messages, census = run_tree ?exceptions dune_files in
    List.iter messages ~f:(fun m -> eprintf "  %s said: %s\n" label (Verdict.one_line m));
    if not (Poly.equal census.mirrors mirrors) then
      List.iter census.mirrors ~f:(fun (library, pairs) ->
          eprintf "  %s paired %s with: %s\n" label library
            (String.concat ~sep:", " (List.map pairs ~f:(fun (m, g) -> m ^ " <- " ^ g))));
    p label (Poly.equal (messages, census.mirrors) ([], mirrors))
  in
  let lacking_format =
    "%s compiles a copy of %s with `%s`, standing for %s, which %s, the stanza compiling that \
     file, does not declare -- the stand-in type-checks where the genuine stanza would not \
     (gh-ocannl-2234); declare it there, or drop it from the stand-in"
  in
  let backend_pair =
    [
      ( "cuda_backend_stub_check",
        [ ("cuda_backend", "(library cuda_backend) of arrayjit/lib/dune") ] );
    ]
  in
  let arm_pair = ("probe_arm", [ ("probe_impl", "(executable probe) of t/dune") ]) in
  refused
    "a stand-in declaring a library its genuine stanza lacks is refused, naming both stanzas (the \
     gh-ocannl-2234 shape)"
    ~format:lacking_format
    ~fragment:
      "arrayjit/vendor_stubs/cudajit/typecheck/dune:2 compiles a copy of \
       arrayjit/lib/cuda_backend.ml with `pprint`, standing for `pprint`, which \
       arrayjit/lib/dune:2"
    (genuine_backend "base cudajit.cuda cudajit.nvrtc stdio"
    :: backend_stand_in "base cudajit_stub stdio pprint");
  mirrored
    "a genuine stanza declaring more than its stand-in, by public names and preprocessors, is \
     accepted"
    ~mirrors:backend_pair
    (genuine_backend ~preprocess:" (preprocess (pps ppx_here ppx_compare -- -flag))"
       "base cudajit.cuda cudajit.nvrtc unix stdio pprint arrayjit.ir"
    :: backend_stand_in
         ~preprocess:" (preprocess (per_module ((pps ppx_here --alert ++foo) cuda_backend)))"
         "base cudajit_stub ir pprint");
  mirrored "a library the genuine stanza names only in a select condition counts as declared"
    ~mirrors:[ arm_pair ]
    (stand_ins
    @ [
        ("t/dune", program ~condition:"cudajit.cuda arrayjit.cuda_backend" ());
        arms_copy ~libraries:"base cudajit_stub cuda_backend_stub_check" ();
      ]);
  refused "a library only a sibling alternative of the arm's own select names is refused"
    ~format:lacking_format ~fragment:"with `unix`, standing for `unix`, which t/dune:1"
    (stand_ins
    @ [
        ( "t/dune",
          "(executable (name probe) (libraries base (select probe_impl.ml from (cudajit.cuda -> \
           probe_impl.cudajit.ml) (unix -> probe_impl.unix.ml) (-> probe_impl.missing.ml))))\n" );
        arms_copy ~libraries:"base cudajit_stub unix" ();
      ]);
  refused "a stand-in re-export the genuine stanza does not make is refused" ~format:lacking_format
    ~fragment:"with `(re_export cudajit_stub)`, standing for"
    (stand_ins @ [ ("t/dune", program ()); arms_copy ~libraries:"base (re_export cudajit_stub)" () ]);
  refused "a package stand-in whose package the genuine stanza never names is refused"
    ~format:lacking_format
    ~fragment:"with `cudajit_stub`, standing for a library of package `cudajit`, which t/dune:1"
    (stand_ins
    @ [
        ("t/dune", program ~condition:"arrayjit.cuda_backend" ());
        arms_copy ~libraries:"base cudajit_stub cuda_backend_stub_check" ();
      ]);
  mirrored "a library an availability-only exception names is excused" ~mirrors:backend_pair
    ~exceptions:[ { stand_in = "cuda_backend_stub_check"; library = "pprint"; reason = "control" } ]
    (genuine_backend "base cudajit.cuda" :: backend_stand_in "base cudajit_stub pprint");
  refused "an availability-only exception that excuses nothing is refused"
    ~format:
      "the availability-only exception for `%s` in stand-in library `%s` (%s) excuses nothing: the \
       library declares no `%s` a genuine stanza it is held to lacks -- delete the exception"
    ~fragment:"exception for `pprint` in stand-in library `cuda_backend_stub_check` (control)"
    ~exceptions:[ { stand_in = "cuda_backend_stub_check"; library = "pprint"; reason = "control" } ]
    (genuine_backend "base cudajit.cuda pprint" :: backend_stand_in "base cudajit_stub pprint");
  refused "a stand-in depending on another library of the stand-in tree is refused"
    ~format:
      "%s declares `%s`, another library of the stand-in tree, which stands for no genuine library \
       -- the declaration cannot be held to a genuine stanza's; depend on the `<package>_stub` or \
       `<backend>_stub_check` library it reaches instead"
    ~fragment:"arrayjit/vendor_stubs/arms/dune:2 declares `probe_helper`"
    (stand_ins @ [ ("t/dune", program ()); arms_copy ~libraries:"base probe_helper" () ]);
  refused "a stand-in copy of a file no stanza compiles is refused"
    ~format:
      "%s compiles `%s`, a copy of %s, and %s stanzas of %s compile that file -- the stand-in's \
       libraries cannot be held to a genuine stanza's; copy a file exactly one stanza compiles"
    ~fragment:"a copy of arrayjit/lib/cuda_backend.ml, and no stanzas of arrayjit/lib compile"
    (("arrayjit/lib/dune", "(library (name other) (modules other))\n")
    :: backend_stand_in "base cudajit_stub");
  refused "a stand-in preprocessor its genuine stanza does not run is refused"
    ~format:
      "%s preprocesses `%s`, a copy of %s, with `%s`, which %s, the stanza compiling that file, \
       does not run over it -- the stand-in type-checks where the genuine stanza would not; add it \
       there, or drop it from the stand-in"
    ~fragment:"preprocesses `cuda_backend`, a copy of arrayjit/lib/cuda_backend.ml, with `ppx_here`"
    (genuine_backend ~preprocess:" (preprocess (pps ppx_compare))" "base cudajit.cuda"
    :: backend_stand_in ~preprocess:" (preprocess (pps ppx_compare ppx_here))" "base cudajit_stub");
  refused "a preprocessing form the scan does not read is refused"
    ~format:
      "%s compiles a copy of %s, and the `(preprocess …)` of %s or %s is not a form this scan \
       reads (%s) -- its preprocessors cannot be compared; spell it as `(pps …)`, `per_module` \
       over `(pps …)`, or `no_preprocessing`"
    ~fragment:"is not a form this scan reads ((staged_pps ppx_here))"
    (genuine_backend "base cudajit.cuda"
    :: backend_stand_in ~preprocess:" (preprocess (staged_pps ppx_here))" "base cudajit_stub");
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
  accepted "a vendor arm copied by a copy_files glob is accepted"
    ~vendor:[ ("t/probe_cuda.ml", Some "arrayjit/vendor_stubs/arms/probe_cuda.ml") ]
    ~ungated:[]
    (stand_ins
    @ [
        ("t/dune", program ~arm:"probe_cuda.ml" ());
        ( "arrayjit/vendor_stubs/arms/dune",
          "(copy_files# (files ../../../t/probe_cuda.{ml,mli}))\n\
           (library (name probe_arm) (modules probe_cuda))\n" );
      ]);
  refused "a default-module stand-in library beside a copy_files glob is refused"
    ~format:
      "%s compiles dune's default module set while a `copy_files` glob copies into %s -- which \
       modules it compiles cannot be read from dune files, so their genuine stanzas cannot be \
       found; list them in `(modules …)`"
    ~fragment:"arrayjit/vendor_stubs/arms/dune:2 compiles dune's default module set"
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
    (stand_ins
    @ [
        ("t/dune", program ());
        ("u/dune", program ~condition:"unix" ());
        arms_copy ~src:"../../../u/probe_impl.cudajit.ml" ();
      ]);
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
  let census = analyze ~fail ~dune_files () in
  (* The tally moves with every new vendor arm, so it stays off the golden; the floor is what says
     the reader is not blind. *)
  eprintf "Vendor-gated select arms: %d (not part of the golden).\n" (List.length census.vendor);
  p_all ~min:1 "every vendor-gated select arm the census finds is compiled by a stand-in copy"
    census.vendor ~f:(fun (_, copy) -> Option.is_some copy);
  (* The library half pairs what the arm half found: a vendor arm's copy left unpaired would be held
     to nothing, silently. *)
  p_all ~min:1 "every vendor-gated arm's stand-in copy is held to the stanza selecting it"
    census.vendor ~f:(fun (arm, copy) ->
      match copy with
      | Some copy -> List.mem census.paired (arm, copy) ~equal:Poly.equal
      | None -> false);
  printf "\nVendor packages with a stand-in: %s\n" (String.concat ~sep:", " census.packages);
  printf "Vendor-gated arrayjit libraries: %s\n" (String.concat ~sep:", " census.backend_libraries);
  printf "\nVendor-gated select arms, each with the stand-in copy that compiles it:\n";
  List.iter census.vendor ~f:(fun (arm, copy) ->
      printf "  %s -> %s\n" arm (Option.value copy ~default:"(no copy compiles it)"));
  printf "\nSelect arms gated on no vendor library (no copy demanded):\n";
  if List.is_empty census.ungated then printf "  (none)\n"
  else List.iter census.ungated ~f:(fun arm -> printf "  %s\n" arm);
  printf "\nStand-in libraries, each copied module with the genuine stanza it is held to:\n";
  List.iter census.mirrors ~f:(fun (library, pairs) ->
      printf "  %s\n" library;
      List.iter pairs ~f:(fun (m, genuine) -> printf "    %s <- %s\n" m genuine));
  printf "\nStand-in libraries compiling no copy (held to nothing): %s\n"
    (String.concat ~sep:", " census.uncopied);
  Verdict.success_trailer ~blank_line:true
    [
      "every vendor-gated select arm is type-checked off its box, against stand-ins declaring \
       nothing their genuine stanzas lack.";
    ];
  Test_scan.Refusal_control_manifest.print "vendor_stub_arms_scan.ml"
