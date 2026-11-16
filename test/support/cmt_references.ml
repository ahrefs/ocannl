(* The compilation units a compiled module actually names (gh-ocannl-2118).

   [unused_compiler_deps] asks whether a stanza's modules reference any unit of a library it
   declares. The answer is read from the TYPED TREE of each [.cmt]/[.cmti]: it is what the compiler
   saw after preprocessing, so a reference a ppx generated counts and one a comment mentions does
   not, and every name in it is already resolved, so [open Ppxlib] followed by [Location.none] reads
   as [Ppxlib], wrapped modules read through their wrapper ([Test_scan.Lexical_scope] heads
   [Test_scan]) and a sibling module of an executable reads as [Dune__exe].

   The import table of the same file is NOT the evidence: it lists every interface the type checker
   loaded, including those reached through another library's signature, so a direct [ppxlib]
   declaration beside a used [test_scan] would read as used there although nothing in the stanza
   names [Ppxlib].

   What counts as a reference (closed): the head persistent unit of every path in a value, module,
   module type, open, class, class type, type constructor, package type, type extension, extension
   rebinding, [with] constraint and pattern-type position, and the type paths of constructors and
   record labels (their declaring type, or an exception's extension path). A position this list
   misses can only make a used library read as unused -- a loud, false finding -- never hide an
   unused one. Two positions are left out because their typed-tree shape differs across the
   supported compilers (5.3 to 5.5): a first-class module's package type and an [[%atomic.loc]]
   field; nothing in the repository references a compiler-reaching library only there.

   Its own library because it reads the compiler's typed tree through [compiler-libs], which puts
   modules under the same names as ppxlib's ([Location], [Longident], [Parsetree]); see
   [emitter_frontier]'s note in this directory's dune file. *)

module Units = Set.Make (String)

let add_path acc path =
  List.fold_left
    (fun acc id -> if Ident.persistent id then Units.add (Ident.name id) acc else acc)
    acc (Path.heads path)

let add_type acc ty =
  match Types.get_desc ty with Types.Tconstr (path, _, _) -> add_path acc path | _ -> acc

let iterator (acc : Units.t ref) =
  let open Typedtree in
  let open Tast_iterator in
  let path p = acc := add_path !acc p in
  (* Constructor and label descriptions moved from [Types] to [Data_types] in OCaml 5.4; reaching
     their fields by type, never by module, keeps this compiling from the 5.3 floor up. *)
  let typ_of ty = acc := add_type !acc ty in
  let expr sub e =
    (match e.exp_desc with
    | Texp_ident (p, _, _) -> path p
    | Texp_construct (_, c, _) -> (
        typ_of c.cstr_res;
        match c.cstr_tag with Cstr_extension (p, _) -> path p | _ -> ())
    | Texp_record { fields; _ } ->
        for i = 0 to Array.length fields - 1 do
          let l, _ = fields.(i) in
          typ_of l.lbl_res
        done
    | Texp_field (_, _, l) | Texp_setfield (_, _, l, _) -> typ_of l.lbl_res
    | Texp_new (p, _, _) -> path p
    | Texp_instvar (p1, p2, _) | Texp_setinstvar (p1, p2, _, _) ->
        path p1;
        path p2
    | Texp_extension_constructor (_, p) -> path p
    | _ -> ());
    default_iterator.expr sub e
  in
  let pat : type k. iterator -> k general_pattern -> unit =
   fun sub p ->
    (match p.pat_desc with
    | Tpat_construct (_, c, _, _) -> (
        typ_of c.cstr_res;
        match c.cstr_tag with Cstr_extension (p, _) -> path p | _ -> ())
    | Tpat_record (fields, _) ->
        (* Indexed, so that each element's type is the one the pattern fixed. *)
        for i = 0 to List.length fields - 1 do
          let _, l, _ = List.nth fields i in
          typ_of l.lbl_res
        done
    | _ -> ());
    List.iter
      (fun (extra, _, _) ->
        match extra with Tpat_type (p, _) | Tpat_open (p, _, _) -> path p | _ -> ())
      p.pat_extra;
    default_iterator.pat sub p
  in
  let typ sub t =
    (match t.ctyp_desc with
    | Ttyp_constr (p, _, _) | Ttyp_class (p, _, _) | Ttyp_open (p, _, _) -> path p
    | _ -> ());
    default_iterator.typ sub t
  in
  let module_expr sub m =
    (match m.mod_desc with Tmod_ident (p, _) -> path p | _ -> ());
    default_iterator.module_expr sub m
  in
  let module_type sub m =
    (match m.mty_desc with Tmty_ident (p, _) | Tmty_alias (p, _) -> path p | _ -> ());
    default_iterator.module_type sub m
  in
  let open_description sub (o : open_description) =
    path (fst o.open_expr);
    default_iterator.open_description sub o
  in
  let with_constraint sub w =
    (match w with Twith_module (p, _) | Twith_modsubst (p, _) -> path p | _ -> ());
    default_iterator.with_constraint sub w
  in
  let type_extension sub te =
    path te.tyext_path;
    default_iterator.type_extension sub te
  in
  let extension_constructor sub ec =
    (match ec.ext_kind with Text_rebind (p, _) -> path p | _ -> ());
    default_iterator.extension_constructor sub ec
  in
  let class_expr sub c =
    (match c.cl_desc with Tcl_ident (p, _, _) -> path p | _ -> ());
    default_iterator.class_expr sub c
  in
  let class_type sub c =
    (match c.cltyp_desc with Tcty_constr (p, _, _) -> path p | _ -> ());
    default_iterator.class_type sub c
  in
  {
    default_iterator with
    expr;
    pat;
    typ;
    module_expr;
    module_type;
    open_description;
    with_constraint;
    type_extension;
    extension_constructor;
    class_expr;
    class_type;
  }

(** The persistent units named by the typed tree of one [.cmt] or [.cmti] file, sorted. [Error] when
    the file cannot be read or holds no complete typed tree (a partial one is what the compiler
    writes for a module that failed to type-check, and reading it would under-count). *)
let referenced_units file =
  match Cmt_format.read_cmt file with
  | exception exn -> Error (Printf.sprintf "%s: unreadable (%s)" file (Printexc.to_string exn))
  | infos -> (
      let acc = ref Units.empty in
      let it = iterator acc in
      match infos.Cmt_format.cmt_annots with
      | Cmt_format.Implementation str ->
          it.structure it str;
          Ok (Units.elements !acc)
      | Cmt_format.Interface sg ->
          it.signature it sg;
          Ok (Units.elements !acc)
      | Cmt_format.Packed _ | Cmt_format.Partial_implementation _ | Cmt_format.Partial_interface _
        ->
          Error (Printf.sprintf "%s: holds no complete typed tree" file))
