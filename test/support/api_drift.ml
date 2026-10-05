open Base
(** Source declarations for editorial review. Implementations retain their bodies: without type
    checking, a body change may change an inferred public type. PPX-generated exports and inferred
    types must still be checked by the editor; this reader does not claim ABI coverage. *)

open Ppxlib.Parsetree

type declaration = { name : string; line : int; text : string }

let binding_names pattern =
  let modules = ref [] in
  let iterator =
    object
      inherit Ppxlib.Ast_traverse.iter as super

      method! pattern pattern =
        (match pattern.ppat_desc with
        | Ppat_unpack { txt = Some name; _ } -> modules := name :: !modules
        | _ -> ());
        super#pattern pattern
    end
  in
  iterator#pattern pattern;
  Dead_export_scan.pattern_names pattern @ !modules

let binding_may_export binding =
  let extension = ref false in
  let iterator =
    object
      inherit Ppxlib.Ast_traverse.iter as super

      method! pattern pattern =
        (match pattern.ppat_desc with Ppat_extension _ -> extension := true | _ -> ());
        super#pattern pattern
    end
  in
  iterator#pattern binding.pvb_pat;
  !extension || not (List.is_empty (binding_names binding.pvb_pat))

(* Every attribute inside a binding pattern: where a pattern PPX would read its input. *)
let pattern_attributes pattern =
  let attributes = ref [] in
  let iterator =
    object
      inherit Ppxlib.Ast_traverse.iter as super

      method! pattern pattern =
        attributes := pattern.ppat_attributes @ !attributes;
        super#pattern pattern
    end
  in
  iterator#pattern pattern;
  !attributes

let canonical_fields = function
  | Sexp.List (head :: children) ->
      let fields, positional =
        List.partition_tf children ~f:(function Sexp.List _ -> true | _ -> false)
      in
      Sexp.List
        ((head :: positional)
        @ List.sort fields ~compare:(fun a b ->
            String.compare (Sexp.to_string a) (Sexp.to_string b)))
  | atom -> atom

(* The stanzas whose fields this reader consumes: module owners and generators. *)
let consumed_heads =
  [ "library"; "executable"; "executables"; "test"; "tests"; "ocamllex"; "menhir" ]

(* The pforms that expand to a file's contents. *)
let read_pforms = [ "%{read:"; "%{read-lines:"; "%{read-strings:" ]

(** Parse [contents], refusing inputs Dune reads from another file. An [(include f)] stanza, an
    [(:include f)] term or a [%{read:f}] form makes stanzas, a module list or configuration depend
    on [f], whose edits this reader would neither see nor attribute; the audited API-root Dune files
    and their history have none (gh-ocannl-1201). *)
let consumed_stanzas dune_path contents =
  let stanzas = Dune_stanza_scan.stanzas contents in
  let refuse form head =
    failwith
      (Printf.sprintf
         "%s: %s in a %s stanza reads another file, which this reader does not follow \
          (gh-ocannl-1201)"
         dune_path form head)
  in
  List.iter stanzas ~f:(fun stanza ->
      let head = Option.value (Dune_stanza_scan.head stanza) ~default:"" in
      if String.equal head "include" then refuse "(include ...)" head
      else if List.mem consumed_heads head ~equal:String.equal then
        (* Only an [(:include ...)] TERM reads a file: a bare [:include] atom is an ordinary
           argument, e.g. to a preprocessing action. *)
        let rec external_input = function
          | Sexp.List (Sexp.Atom ":include" :: _) -> Some ":include"
          | Sexp.List terms -> List.find_map terms ~f:external_input
          | Sexp.Atom atom ->
              Option.some_if
                (List.exists read_pforms ~f:(fun pform -> String.is_substring atom ~substring:pform))
                atom
        in
        Option.iter (external_input stanza) ~f:(fun form -> refuse form head));
  stanzas

type ownership = {
  generators : (string * string * Sexp.t) list;  (** generator input, generated [.ml], config *)
  public_modules : Set.M(String).t;
      (** lowercase modules of public libraries, [private_modules] included: privacy does not keep a
          module's declarations out of the public API, which can [include] or alias it *)
}

let ownership ~paths (dune_path, contents) =
  let directory = Stdlib.Filename.dirname dune_path in
  let path name = directory ^ "/" ^ name in
  let stanzas = consumed_stanzas dune_path contents in
  let rec select_inputs = function
    | Sexp.List (Sexp.Atom "select" :: Sexp.Atom output :: Sexp.Atom "from" :: clauses) as config ->
        if Option.is_none (Dead_export_scan.module_name_of_source output) then
          failwith
            (Printf.sprintf
               "unsupported select target %s in %s: only implementation (.ml) targets are read \
                (gh-ocannl-1201)"
               output dune_path);
        List.map clauses ~f:(function
          | Sexp.List terms -> (
              match List.rev terms with
              | Sexp.Atom input :: Sexp.Atom "->" :: _ -> (path input, path output, config)
              | _ -> failwith ("unsupported select clause in " ^ dune_path))
          | _ -> failwith ("unsupported select clause in " ^ dune_path))
    | Sexp.List children -> List.concat_map children ~f:select_inputs
    | Sexp.Atom _ -> []
  in
  let generators =
    List.concat_map stanzas ~f:(fun stanza ->
        let spec =
          match Dune_stanza_scan.head stanza with
          | Some "ocamllex" ->
              let modules =
                match Dune_stanza_scan.field stanza "modules" with
                | Some modules -> modules
                | None -> ( match stanza with Sexp.List (_ :: modules) -> modules | _ -> [])
              in
              Some (".mll", modules, None)
          | Some "menhir" ->
              let modules = Option.value (Dune_stanza_scan.field stanza "modules") ~default:[] in
              let output =
                match Dune_stanza_scan.field stanza "merge_into" with
                | Some [ Sexp.Atom name ] -> Some name
                | None -> None
                | Some _ -> failwith ("unsupported menhir merge_into in " ^ dune_path)
              in
              Some (".mly", modules, output)
          | _ -> None
        in
        match spec with
        | None -> []
        | Some (suffix, modules, output) ->
            List.map modules ~f:(function
              | Sexp.Atom name
                when Option.is_some (Dead_export_scan.module_name_of_source (name ^ ".ml")) ->
                  ( path (name ^ suffix),
                    path (Option.value output ~default:name ^ ".ml"),
                    canonical_fields stanza )
              | _ -> failwith ("unsupported generator module set in " ^ dune_path)))
    @ List.concat_map stanzas ~f:select_inputs
  in
  let directory_modules =
    List.filter_map paths ~f:(fun p ->
        if String.equal (Stdlib.Filename.dirname p) directory then
          Dead_export_scan.module_name_of_source p
        else None)
    @ List.map generators ~f:(fun (_, output, _) ->
        Option.value_exn (Dead_export_scan.module_name_of_source output))
    |> List.map ~f:String.lowercase
    |> List.dedup_and_sort ~compare:String.compare
  in
  let owners =
    List.filter stanzas ~f:(fun s ->
        List.mem
          [ "library"; "executable"; "executables"; "test"; "tests" ]
          (Option.value (Dune_stanza_scan.head s) ~default:"")
          ~equal:String.equal)
  in
  let public_modules =
    List.filter owners ~f:(fun owner ->
        Option.equal String.equal (Dune_stanza_scan.head owner) (Some "library")
        && not (List.is_empty (Dune_stanza_scan.public_names owner)))
    |> List.concat_map ~f:(Dune_stanza_scan.modules_of ~directory_modules owners)
    |> List.map ~f:String.lowercase
    |> Set.of_list (module String)
  in
  { generators; public_modules }

let derived_module_inputs ~paths dunes =
  let present = Set.of_list (module String) paths in
  List.concat_map dunes ~f:(fun dune ->
      let { generators; public_modules } = ownership ~paths dune in
      List.filter_map generators ~f:(fun (input, output, config) ->
          let name =
            Option.value_exn (Dead_export_scan.module_name_of_source output) |> String.lowercase
          in
          let interface = String.chop_suffix_exn output ~suffix:".ml" ^ ".mli" in
          Option.some_if
            (Set.mem present input && Set.mem public_modules name && not (Set.mem present interface))
            (input, output, config)))

let derived_inputs ~paths dunes =
  derived_module_inputs ~paths dunes |> List.map ~f:(fun (input, _, _) -> input)

let publication_inputs ?(paths = []) ~source contents =
  let libraries =
    consumed_stanzas source contents
    |> List.filter_map ~f:(fun stanza ->
        if
          Option.equal String.equal (Dune_stanza_scan.head stanza) (Some "library")
          && not (List.is_empty (Dune_stanza_scan.public_names stanza))
        then
          let rec re_exports = function
            | Sexp.List (Sexp.Atom "re_export" :: _) as term -> [ term ]
            | Sexp.List terms -> List.concat_map terms ~f:re_exports
            | Sexp.Atom _ -> []
          in
          let fields =
            (* Literal configuration evidence, not a Dune evaluator. Ordinary dependencies do not
               describe declarations, but a [re_export] publishes its library to every user, at any
               depth of the dependency list; accepted select inputs are retained separately
               below. *)
            (match stanza with Sexp.List (_ :: fields) -> fields | _ -> [])
            |> List.filter_map ~f:(fun field ->
                match Dune_stanza_scan.head field with
                | Some "synopsis" -> None
                | Some "libraries" -> (
                    match re_exports field with
                    | [] -> None
                    | terms -> Some (Sexp.List (Sexp.Atom "libraries" :: terms)))
                | _ -> Some field)
            |> List.sort ~compare:(fun a b -> String.compare (Sexp.to_string a) (Sexp.to_string b))
          in
          Some
            {
              name = "public library " ^ String.concat ~sep:"," (Dune_stanza_scan.names_of stanza);
              line = 1;
              text = Sexp.to_string_hum (Sexp.List (Sexp.Atom "library" :: fields));
            }
        else None)
  in
  let configurations =
    derived_module_inputs ~paths [ (source, contents) ]
    |> List.map ~f:(fun (_, output, config) ->
        { name = "derived module " ^ output; line = 1; text = Sexp.to_string_hum config })
  in
  (* Independent Dune stanza order has no OCaml name-resolution meaning. *)
  List.dedup_and_sort (libraries @ configurations) ~compare:(fun a b ->
      match String.compare a.name b.name with 0 -> String.compare a.text b.text | order -> order)

let sources ?(dunes = []) paths =
  (* Ordinary sources follow the dead-export census. A module a library declares private stays: a
     public module can [include] or alias it, and its edits must then remain evidence. *)
  List.filter paths ~f:(fun path ->
      Dead_export_scan.in_scan_root path && String.is_suffix path ~suffix:".mli")
  @ List.filter (Dead_export_scan.implicit_implementations paths) ~f:(fun source ->
      Option.is_some (Dead_export_scan.module_name_of_source source))
  @ derived_inputs ~paths dunes
  @ List.filter_map dunes ~f:(fun (path, contents) ->
      Option.some_if (not (List.is_empty (publication_inputs ~paths ~source:path contents))) path)
  |> List.dedup_and_sort ~compare:String.compare

let declarations ?(paths = []) ~source contents =
  if String.equal (Stdlib.Filename.basename source) "dune" then
    publication_inputs ~paths ~source contents
  else if String.is_suffix source ~suffix:".mll" || String.is_suffix source ~suffix:".mly" then
    (* This is deliberately a review entry for a generator INPUT. Generating and typechecking a
       historical module would require its historical dependency tree and toolchain. Keep all input
       changes visible, without claiming to reconstruct its generated/inferred interface. *)
    [ { name = "generated module input[0]"; line = 1; text = contents } ]
  else
    let counts = Hashtbl.create (module String) in
    let entry name loc text =
      let ordinal = Hashtbl.find_or_add counts name ~default:(fun () -> 0) in
      Hashtbl.set counts ~key:name ~data:(ordinal + 1);
      {
        name = Printf.sprintf "%s[%d]" name ordinal;
        line = loc.Ppxlib.Location.loc_start.pos_lnum;
        text;
      }
    in
    let type_names declarations =
      "type " ^ String.concat ~sep:"," (List.map declarations ~f:(fun t -> t.ptype_name.txt))
    in
    let signature_item item =
      let name =
        match item.psig_desc with
        | Psig_value v -> "val " ^ v.pval_name.txt
        | Psig_type (_, ts) | Psig_typesubst ts -> type_names ts
        | Psig_typext t -> "type extension " ^ Ppxlib.Longident.name t.ptyext_path.txt
        | Psig_exception e -> "exception " ^ e.ptyexn_constructor.pext_name.txt
        | Psig_module m -> "module " ^ Option.value m.pmd_name.txt ~default:"_"
        | Psig_recmodule ms ->
            "module rec " ^ String.concat ~sep:"," (List.filter_map ms ~f:(fun m -> m.pmd_name.txt))
        | Psig_modtype m | Psig_modtypesubst m -> "module type " ^ m.pmtd_name.txt
        | Psig_modsubst m -> "module substitution " ^ m.pms_name.txt
        | Psig_include _ -> "include"
        | Psig_class cs ->
            "class " ^ String.concat ~sep:"," (List.map cs ~f:(fun c -> c.pci_name.txt))
        | Psig_class_type cs ->
            "class type " ^ String.concat ~sep:"," (List.map cs ~f:(fun c -> c.pci_name.txt))
        | Psig_extension ((name, _), _) -> "extension " ^ name.txt
        | Psig_open _ | Psig_attribute _ -> "environment"
      in
      entry name item.psig_loc (Stdlib.Format.asprintf "%a" Ppxlib.Pprintast.signature [ item ])
    in
    let structure_item item =
      let name =
        match item.pstr_desc with
        | Pstr_value (_, bindings) ->
            "let "
            ^ String.concat ~sep:","
                (List.concat_map bindings ~f:(fun b -> binding_names b.pvb_pat))
        | Pstr_primitive v -> "external " ^ v.pval_name.txt
        | Pstr_type (_, ts) -> type_names ts
        | Pstr_typext t -> "type extension " ^ Ppxlib.Longident.name t.ptyext_path.txt
        | Pstr_exception e -> "exception " ^ e.ptyexn_constructor.pext_name.txt
        | Pstr_module m -> "module " ^ Option.value m.pmb_name.txt ~default:"_"
        | Pstr_recmodule ms ->
            "module rec " ^ String.concat ~sep:"," (List.filter_map ms ~f:(fun m -> m.pmb_name.txt))
        | Pstr_modtype m -> "module type " ^ m.pmtd_name.txt
        | Pstr_include _ -> "include"
        | Pstr_class cs ->
            "class " ^ String.concat ~sep:"," (List.map cs ~f:(fun c -> c.pci_name.txt))
        | Pstr_class_type cs ->
            "class type " ^ String.concat ~sep:"," (List.map cs ~f:(fun c -> c.pci_name.txt))
        | Pstr_extension ((name, _), _) -> "extension " ^ name.txt
        | Pstr_open _ | Pstr_attribute _ -> "environment"
        | Pstr_eval _ -> "evaluation"
      in
      entry name item.pstr_loc (Stdlib.Format.asprintf "%a" Ppxlib.Pprintast.structure [ item ])
    in
    let lexbuf = Lexing.from_string contents in
    Ppxlib.Location.init lexbuf source;
    (* Keep attributes that can generate or alter public surface, but discard prose. *)
    let is_doc a = List.mem [ "ocaml.doc"; "ocaml.text" ] a.attr_name.txt ~equal:String.equal in
    (* An anonymous item exports nothing by itself, so its edits are pruned; only an attribute PPX
       could make it export, and no such producer exists in the API roots or their history. This
       reader does not interpret attribute semantics, so rather than drop an attribute input or
       guess its effect, it refuses any non-documentation attribute on such an item or inside its
       binding pattern (gh-ocannl-1201). Attributes inside expressions ([@inline], ...) are not
       export positions and pass. *)
    let anonymous what loc attrs =
      Option.iter
        (List.find attrs ~f:(fun a -> not (is_doc a)))
        ~f:(fun a ->
          failwith
            (Printf.sprintf
               "%s:%d: attribute %s on an anonymous %s: attribute PPXs exporting from anonymous \
                items are outside this source inventory (gh-ocannl-1201)"
               loc.Ppxlib.Location.loc_start.pos_fname loc.loc_start.pos_lnum a.attr_name.txt what))
    in
    let strip_docs =
      object
        inherit Ppxlib.Ast_traverse.map as super
        val mutable prune_nonexports = true

        (* An extension may turn any payload into exported declarations. Keep its input visible;
           dropping a payload evaluation would silently turn [%%publish earlier] and [%%publish
           later] into the same empty extension. *)
        method! extension extension =
          let saved = prune_nonexports in
          prune_nonexports <- false;
          Exn.protect
            ~f:(fun () -> super#extension extension)
            ~finally:(fun () -> prune_nonexports <- saved)

        (* Attributes, like extensions, can consume an arbitrary structure payload to generate
           exports. Only documentation attributes are discarded by the enclosing filters. *)
        method! attribute attribute =
          let saved = prune_nonexports in
          prune_nonexports <- false;
          Exn.protect
            ~f:(fun () -> super#attribute attribute)
            ~finally:(fun () -> prune_nonexports <- saved)

        method! attributes attrs = super#attributes (List.filter attrs ~f:(fun a -> not (is_doc a)))

        method! signature items =
          super#signature
            (List.filter items ~f:(fun item ->
                 match item.psig_desc with Psig_attribute a -> not (is_doc a) | _ -> true))

        (* A pruned item is still walked, discarding the result, so that the refusal reaches the
           attributed anonymous items nested in it ([module _ = struct let () = ... [@@x] end]). *)
        method! structure items =
          super#structure
            (List.filter_map items ~f:(fun item ->
                 match item.pstr_desc with
                 | Pstr_eval (_, attrs) when prune_nonexports ->
                     anonymous "evaluation" item.pstr_loc attrs;
                     ignore (super#structure_item item : structure_item);
                     None
                 | Pstr_module { pmb_name = { txt = None; _ }; pmb_attributes; pmb_loc; _ }
                   when prune_nonexports ->
                     anonymous "module binding" pmb_loc pmb_attributes;
                     ignore (super#structure_item item : structure_item);
                     None
                 | Pstr_value (recursive, bindings) when prune_nonexports ->
                     let bindings =
                       List.filter bindings ~f:(fun b ->
                           binding_may_export b
                           ||
                           (anonymous "value binding" b.pvb_loc
                              (b.pvb_attributes @ pattern_attributes b.pvb_pat);
                            ignore (super#value_binding b : value_binding);
                            false))
                     in
                     if List.is_empty bindings then None
                     else Some { item with pstr_desc = Pstr_value (recursive, bindings) }
                 | Pstr_attribute a when is_doc a -> None
                 | _ -> Some item))
      end
    in
    if String.is_suffix source ~suffix:".mli" then
      Ppxlib.Parse.interface lexbuf |> strip_docs#signature |> List.map ~f:signature_item
    else Ppxlib.Parse.implementation lexbuf |> strip_docs#structure |> List.map ~f:structure_item

let changes before after =
  let map declarations =
    Map.of_alist_exn (module String) (List.map declarations ~f:(fun d -> (d.name, d)))
  in
  let before_map = map before and after_map = map after in
  (* Compare the order of surviving entries: additions/removals do not move every following
     declaration, but moving across an open or another declaration may change name resolution. *)
  let positions declarations other =
    List.filter declarations ~f:(fun d -> Map.mem other d.name)
    |> List.mapi ~f:(fun index d -> (d.name, index))
    |> Map.of_alist_exn (module String)
  in
  let before_positions = positions before after_map
  and after_positions = positions after before_map in
  Map.merge before_map after_map ~f:(fun ~key -> function
    | `Both (a, b)
      when String.equal a.text b.text
           && Int.equal (Map.find_exn before_positions key) (Map.find_exn after_positions key) ->
        None
    | `Both (a, b) -> Some (Some a, Some b)
    | `Left a -> Some (Some a, None)
    | `Right b -> Some (None, Some b))
  |> Map.data

type line_edit = Same of string | Removed of string | Added of string

(* Above this many comparison cells the middle that differs is reported as wholly replaced: still a
   correct edit script, just not a minimal one, and the memory stays bounded. *)
let max_diff_cells = 1_000_000

(** A line edit script from [before] to [after]: common ends trimmed, then a longest common
    subsequence over the middle. *)
let line_edits before after =
  let before = Array.of_list before and after = Array.of_list after in
  let n = Array.length before and m = Array.length after in
  let prefix = ref 0 in
  while !prefix < n && !prefix < m && String.equal before.(!prefix) after.(!prefix) do
    Int.incr prefix
  done;
  let suffix = ref 0 in
  while
    !suffix < n - !prefix
    && !suffix < m - !prefix
    && String.equal before.(n - 1 - !suffix) after.(m - 1 - !suffix)
  do
    Int.incr suffix
  done;
  let same lo hi source = List.init (hi - lo) ~f:(fun i -> Same source.(lo + i)) in
  let rows = n - !prefix - !suffix and cols = m - !prefix - !suffix in
  let middle =
    let a i = before.(!prefix + i) and b j = after.(!prefix + j) in
    if rows * cols > max_diff_cells then
      List.init rows ~f:(fun i -> Removed (a i)) @ List.init cols ~f:(fun j -> Added (b j))
    else
      (* [lcs.(i).(j)]: the longest common subsequence of the suffixes from [i] and [j]. *)
      let lcs = Array.make_matrix ~dimx:(rows + 1) ~dimy:(cols + 1) 0 in
      for i = rows - 1 downto 0 do
        for j = cols - 1 downto 0 do
          lcs.(i).(j) <-
            (if String.equal (a i) (b j) then lcs.(i + 1).(j + 1) + 1
             else Int.max lcs.(i + 1).(j) lcs.(i).(j + 1))
        done
      done;
      let rec walk acc i j =
        if i = rows then List.rev_append acc (List.init (cols - j) ~f:(fun k -> Added (b (j + k))))
        else if j = cols then
          List.rev_append acc (List.init (rows - i) ~f:(fun k -> Removed (a (i + k))))
        else if String.equal (a i) (b j) then walk (Same (a i) :: acc) (i + 1) (j + 1)
        else if lcs.(i + 1).(j) >= lcs.(i).(j + 1) then walk (Removed (a i) :: acc) (i + 1) j
        else walk (Added (b j) :: acc) i (j + 1)
      in
      walk [] 0 0
  in
  same 0 !prefix before @ middle @ same (n - !suffix) n before

(** The report lines of one checklist entry. Both sides print in full by default: that is the
    evidence. With [~context], an entry present on both sides keeps both attribution headers but
    prints only its changed lines ([-]/[+]), [context] unchanged lines around each ([  ]), and a
    count of every other unchanged line ([~]). *)
let render ?context (old, fresh) =
  let header prefix d = Printf.sprintf "%s %s (line %d)" prefix d.name d.line in
  let side prefix = function
    | None -> []
    | Some d ->
        header prefix d :: List.map (String.split_lines d.text) ~f:(fun l -> prefix ^ " " ^ l)
  in
  match (context, old, fresh) with
  | Some context, Some before, Some after ->
      let edits =
        Array.of_list (line_edits (String.split_lines before.text) (String.split_lines after.text))
      in
      let len = Array.length edits in
      let changed i = match edits.(i) with Same _ -> false | Removed _ | Added _ -> true in
      (* [near.(i)]: within [context] lines of a changed line, found by one pass each way. *)
      let near = Array.create ~len false in
      let mark order =
        let last = ref None in
        List.iter order ~f:(fun i ->
            if changed i then last := Some i;
            Option.iter !last ~f:(fun c -> if Int.abs (i - c) <= context then near.(i) <- true))
      in
      mark (List.range 0 len);
      mark (List.rev (List.range 0 len));
      let omitted count =
        Printf.sprintf "~ %d unchanged line%s" count (if count = 1 then "" else "s")
      in
      let flush skipped acc = if skipped > 0 then omitted skipped :: acc else acc in
      let rec lines acc skipped i =
        if i = len then List.rev (flush skipped acc)
        else if not near.(i) then lines acc (skipped + 1) (i + 1)
        else
          let line =
            match edits.(i) with Same l -> "  " ^ l | Removed l -> "- " ^ l | Added l -> "+ " ^ l
          in
          lines (line :: flush skipped acc) 0 (i + 1)
      in
      let body =
        if Array.exists edits ~f:(function Same _ -> false | Removed _ | Added _ -> true) then
          lines [] 0 0
        else [ omitted len ^ "; only the position among surviving entries changed" ]
      in
      header "-" before :: header "+" after :: body
  | _ -> side "-" old @ side "+" fresh
