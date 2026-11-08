(* The one relationship the dune-marker grammar rests on and nothing used to state (gh-ocannl-689).

   [Dune_stanza_scan.marker_backends] is the closed vocabulary of the [; ocannl-backend: <word> --
   <reason>] stanza marker, and its closedness is load-bearing: it is what makes [; ocannl-backend:
   metl -- ...] fail instead of reading as a truthful exemption. But the list is written out as
   text, deliberately -- the scanning tests link [arrayjit.utils] and the source scanners, and
   dragging the backend closure into a check that reads dune files would be a bad trade -- so
   nothing in the scanner relates it to the backends OCANNL actually has.

   Both drifts are silent, and the one that matters is not the one it looks like. Remove or rename a
   backend and markers naming the dead one keep passing. Add one, and the CORRECT marker for it is
   rejected as malformed; the author's fix is then to reach for [none], which is a lie the grammar
   accepts. So the failure mode is not a red build someone fixes by editing a list, it is pressure
   toward the least honest classification available.

   This test is where the two lists meet, and it exists as its own executable because that is the
   whole cost being managed: it links the backend closure so that no scanning test has to. It starts
   no context, opens no device and compiles nothing -- [Backends.all_of_backend] is a derived
   constant.

   It has since become the place for every relationship that needs the closure, for the same reason:
   the second one below relates the backends to the capability facts their modules state, the third
   to what the configuration reader writes for each of them, and the fourth holds the compiler's
   sources to asking those facts rather than a backend's name. *)

open Base
module Backends = Context.Backends
module Scan = Test_scan.Dune_stanza_scan

let () =
  let sorted l = List.sort l ~compare:String.compare in
  (* Plus [none], which is not a backend: it says the run does not depend on the configured backend
     at all. Every other word must name one. *)
  let expected = sorted ("none" :: List.map Backends.all_of_backend ~f:Backends.backend_name) in
  let admitted = sorted Scan.marker_backends in
  (* Compared as sorted lists rather than as sets, so a word repeated on either side is a mismatch
     too: a vocabulary with a duplicate in it is not the set it is standing in for. *)
  let agree = List.equal String.equal admitted expected in
  if not agree then
    (* The claim is a bare boolean so that the golden stays fixed as backends come and go; the two
       lists go to stderr, where a failing run shows which way they parted. *)
    Stdio.eprintf "the marker admits [%s]; the backends OCANNL has, plus none, are [%s]\n"
      (String.concat ~sep:"; " admitted)
      (String.concat ~sep:"; " expected);
  Verdict.p "the ocannl-backend marker vocabulary is exactly OCANNL's backends plus none" agree

(* The second relationship, gh-ocannl-706 and gh-ocannl-1723. What a compiler decision asks of a
   backend -- whether its [Grid] loops bind a hardware grid or render on the host pool, whether it
   seeds the untuned plan, how it times -- is the backend's [Ir.Backend_intf.backend_facts], stated
   by its module. It used to be read off the backend's NAME by substring ([backend_is_cpu] was
   "contains cc"), so a backend the substrings were never taught read as neither CPU nor GPU, and
   every caller phrased [if on_gpu then ... else ...] took the CPU branch for it; [hipcc] would have
   read as both. Exactly-one is now the type's ([grid_execution] has two constructors), so what is
   left to hold is that every backend's facts are reachable from its name, consistent with one
   another, and that a name is parsed, never classified. *)
let () =
  let module BI = Ir.Backend_intf in
  (* Every name [get_backend] accepts -- the canonical ones and the deprecated aliases -- reaches
     exactly the facts the module of the backend it parses to states. *)
  Verdict.p_all "every accepted backend name reaches the facts its backend's module states"
    Backends.accepted_names ~f:(fun (name, backend) ->
      BI.equal_backend_facts (Context.backend_facts_of_name name) (Backends.backend_facts backend));
  let inconsistent =
    List.filter_map Backends.all_of_backend ~f:(fun b ->
        match BI.facts_inconsistencies (Backends.backend_facts b) with
        | [] -> None
        | broken -> Some (Backends.backend_name b, broken))
  in
  List.iter inconsistent ~f:(fun (name, broken) ->
      Stdio.eprintf "%s states inconsistent facts: %s\n" name (String.concat ~sep:"; " broken));
  Verdict.p_empty "every backend OCANNL has states facts consistent with one another"
    ~over:Backends.all_of_backend inconsistent;
  (* Names the substring predicates classified: [hipcc] as CPU and GPU at once, [cuda_cc] and
     [metal_cc] likewise, [vulkan] as neither. Drawn from outside the corpus, because every backend
     name today parses: a control from the corpus would encode the absence of the violating
     shape. *)
  let refused name =
    match Context.backend_facts_of_name name with
    | (_ : BI.backend_facts) -> false
    | exception Invalid_argument _ -> true
  in
  Verdict.p_all "a name that is not a backend's is refused, never classified by what it contains"
    [ "hipcc"; "cuda_cc"; "metal_cc"; "vulkan" ]
    ~f:refused;
  (* The consistency check's own control: each relation, broken alone on a backend's facts, is
     reported. *)
  let breaks =
    [
      { BI.cc_facts with workgroup_shared_placement = true };
      { BI.cc_facts with seeds_untuned_default = true };
      { BI.cc_facts with queued_timing = Queued_deep };
      { BI.cc_facts with retries_contended_window = true };
      { BI.cc_facts with node_ident_prefix = "t_" };
      { BI.metal_facts with node_ident_prefix = "" };
    ]
  in
  Verdict.p_none "and a record breaking any one relation is reported" breaks ~f:(fun facts ->
      List.is_empty (BI.facts_inconsistencies facts))

(* The third relationship, gh-ocannl-1581. [Config_reads] is what [ocannl_read_config] writes for a
   backend setting -- the canonical name a per-backend golden is named after, the extension an
   emitted-code artifact carries -- and [codegen_text_inventory] names the backend a golden's
   condition means by running it over every backend; [Test_utils.Generated] finds a routine's
   artifact through it. It links [base] alone, so its two tables restate what the backends state,
   and this is where each is held equal to its source rather than to a second copy of itself.

   Both drifts would be silent. An alias added to [Backends.accepted_names] alone resolves, in the
   reader, to itself -- a name no backend has -- so a log golden is named after a backend no run
   uses. And an extension the reader disagrees with the backend on files a backend's kernel
   snapshots under another dialect, and points [Generated] at a file the backend never writes. The
   extension's source is the backend's [source_extension] fact, which its artifact writer appends,
   so a backend renaming what it writes moves the fact this compares. *)
module BI = Ir.Backend_intf

(* The entries of a parse table that are not a backend's own name: the deprecated aliases, each with
   the canonical name of the backend it parses to. *)
let aliases_of accepted =
  List.filter_map accepted ~f:(fun (name, backend) ->
      let canonical = Backends.backend_name backend in
      Option.some_if (not (String.equal name canonical)) (name, canonical))

let spelled pairs =
  List.sort ~compare:String.compare
    (List.map pairs ~f:(fun (alias, canonical) -> alias ^ " -> " ^ canonical))

(* Compared as sorted lists, so a pair repeated on either side is a mismatch too. *)
let aliases_agree ~accepted ~reader =
  List.equal String.equal (spelled (aliases_of accepted)) (spelled reader)

(* The reader's extension for [backend], dot added, against the one its facts state. *)
let extension_agrees ~code_extension ~facts backend =
  Option.equal String.equal
    (Option.map (code_extension (Backends.backend_name backend)) ~f:(fun e -> "." ^ e))
    (Some facts.BI.source_extension)

let () =
  let names = List.map Backends.all_of_backend ~f:Backends.backend_name in
  let reader = Config_reads.deprecated_aliases in
  let agree = aliases_agree ~accepted:Backends.accepted_names ~reader in
  if not agree then
    Stdio.eprintf "the reader's aliases are [%s]; the parse table's are [%s]\n"
      (String.concat ~sep:"; " (spelled reader))
      (String.concat ~sep:"; " (spelled (aliases_of Backends.accepted_names)));
  Verdict.p
    "the reader's deprecated aliases are exactly the parse table's entries that are not a \
     backend's own name"
    agree;
  Verdict.p_all "the reader writes every backend OCANNL has under its own name" names
    ~f:(fun name ->
      Option.equal String.equal (Config_reads.written ~key:"backend" name) (Some name));
  Verdict.p_all "the reader's emitted-code extension is the one each backend's facts state"
    Backends.all_of_backend ~f:(fun b ->
      extension_agrees ~code_extension:Config_reads.code_extension ~facts:(Backends.backend_facts b)
        b);
  (* [Generated]'s extension, through every name the parse table accepts and in either case, is the
     facts' of the backend the name parses to; a name that is not one is refused. *)
  let generated_extension name =
    match Test_utils.Generated.extension_of_backend name with
    | ext -> Some ext
    | exception Invalid_argument _ -> None
  in
  Verdict.p_all
    "Generated reads every accepted backend name's artifacts under its backend's own extension"
    Backends.accepted_names ~f:(fun (name, backend) ->
      let ext = Some (Backends.backend_facts backend).source_extension in
      Option.equal String.equal (generated_extension name) ext
      && Option.equal String.equal (generated_extension (String.uppercase name)) ext);
  (* The controls. Names that contain a backend's name, which the substring test read as that
     backend, and one that contains none, which it read as C. *)
  Verdict.p_none "Generated refuses a name that is no backend's, whatever it contains"
    [ "xcuda"; "hipcc"; "metal_cc"; "vulkan" ] ~f:(fun name ->
      Option.is_some (generated_extension name));
  Verdict.p "control: a name that is no backend has no extension in the reader"
    (Option.is_none (Config_reads.code_extension "vulkan"));
  Verdict.p "control: an alias added to the parse table alone breaks the alias agreement"
    (not (aliases_agree ~accepted:(("xcc", Backends.Cc) :: Backends.accepted_names) ~reader));
  Verdict.p "control: and so does an alias the reader resolves to another backend"
    (not
       (aliases_agree ~accepted:Backends.accepted_names
          ~reader:(List.map reader ~f:(fun (alias, _) -> (alias, "cuda")))));
  Verdict.p "control: a reader disagreeing on one backend's extension breaks the agreement"
    (not
       (extension_agrees
          ~code_extension:(function "cuda" -> Some "c" | name -> Config_reads.code_extension name)
          ~facts:(Backends.backend_facts Backends.Cuda)
          Backends.Cuda));
  Verdict.p "control: and so does a backend writing under another extension"
    (not
       (extension_agrees ~code_extension:Config_reads.code_extension
          ~facts:{ BI.metal_facts with source_extension = ".msl" }
          Backends.Metal))

(* The fourth relationship, gh-ocannl-1723's ratchet: no compiler decision tests a backend NAME.
   Every string constant in the compiler's sources -- [arrayjit/lib], [lib], [tensor], in an
   expression or a pattern -- that spells a name [Backends.get_backend] accepts sits in an exemption
   below that counts it and says why it is not a decision. The vocabulary is
   [Backends.accepted_names], read, not restated, so a new backend or alias joins the check by
   joining the parse table.

   BOUNDARY. String constants only, by exact value. A name assembled at run time, compared through a
   variable, or spelled in another case is past it; so is a decision keyed on something else that
   stands for the name. The constant is the shape the substring predicates had, and the one a
   regression would most plausibly bring back.

   An exemption is a file and the innermost structure-level binding the constants sit in, with how
   many: one more constant there is refused as unlisted, one fewer as a stale count. A
   dune-[select]ed source ([x.ml] beside [x.missing.ml]) is a copy of one of its alternatives and is
   read through them. *)
module Literals = struct
  open Ppxlib

  type site = { file : string; binding : string; line : int; literal : string }

  let rec binding_name pat =
    match pat.ppat_desc with
    | Ppat_var { txt; _ } -> txt
    | Ppat_constraint (p, _) -> binding_name p
    | Ppat_construct ({ txt = Lident "()"; _ }, None) -> "()"
    | _ -> "<pattern>"

  let sites ~vocabulary ~file content =
    let found = ref [] in
    let record binding (loc : Location.t) s =
      if List.mem vocabulary s ~equal:String.equal then
        found := { file; binding; line = loc.loc_start.pos_lnum; literal = s } :: !found
    in
    let visitor binding =
      object
        inherit Ast_traverse.iter as super

        method! expression e =
          (match e.pexp_desc with
          | Pexp_constant (Pconst_string (s, _, _)) -> record binding e.pexp_loc s
          | _ -> ());
          super#expression e

        method! pattern p =
          (match p.ppat_desc with
          | Ppat_constant (Pconst_string (s, _, _)) -> record binding p.ppat_loc s
          | _ -> ());
          super#pattern p
      end
    in
    let rec items ~outer =
      List.iter ~f:(fun item ->
          match item.pstr_desc with
          | Pstr_value (_, vbs) ->
              List.iter vbs ~f:(fun vb -> (visitor (binding_name vb.pvb_pat))#value_binding vb)
          | Pstr_module mb -> module_expr ~outer mb.pmb_expr
          | Pstr_recmodule mbs -> List.iter mbs ~f:(fun mb -> module_expr ~outer mb.pmb_expr)
          | Pstr_include incl -> module_expr ~outer incl.pincl_mod
          | _ -> (visitor outer)#structure_item item)
    and module_expr ~outer me =
      match me.pmod_desc with
      | Pmod_structure s -> items ~outer s
      | Pmod_functor (_, body) -> module_expr ~outer body
      | Pmod_constraint (me, _) -> module_expr ~outer me
      | Pmod_apply (f, arg) ->
          module_expr ~outer f;
          module_expr ~outer arg
      | _ -> (visitor outer)#module_expr me
    in
    items ~outer:"<top>" (Parse.implementation (Lexing.from_string content));
    List.rev !found

  type exemption = { e_file : string; e_binding : string; e_count : int }

  (* The unlisted constants and the exemptions whose count no longer matches. *)
  let check ~exemptions sites =
    let key file binding = file ^ " " ^ binding in
    let counts =
      List.fold sites
        ~init:(Map.empty (module String))
        ~f:(fun m s ->
          Map.update m (key s.file s.binding) ~f:(fun c -> 1 + Option.value c ~default:0))
    in
    let exempt s =
      List.exists exemptions ~f:(fun e ->
          String.equal e.e_file s.file && String.equal e.e_binding s.binding)
    in
    let unlisted = List.filter sites ~f:(Fn.non exempt) in
    let stale =
      List.filter exemptions ~f:(fun e ->
          Option.value (Map.find counts (key e.e_file e.e_binding)) ~default:0 <> e.e_count)
    in
    (unlisted, stale)
end

(* Each entry: where, how many, and why the constant is not a compiler decision. *)
let exemptions =
  let e e_file e_binding e_count (_why : string) = { Literals.e_file; e_binding; e_count } in
  [
    e "arrayjit/lib/backends.ml" "accepted_names" 7
      "the parse table: the one place a backend name is tested, reading the backend setting";
    e "arrayjit/lib/backends.ml" "get_backend" 1 "the [backend] setting's default, which it parses";
    e "arrayjit/lib/backends.ml" "backend_name" 5
      "the variant's inverse, for messages and cache keys";
    e "arrayjit/lib/cc_backend.ml" "name" 1 "the backend's own name";
    e "arrayjit/lib/cc_backend.ml" "compiler_command" 1
      "not a backend name: the C compiler command [ocamlc -config] falls back to";
    e "arrayjit/lib/cuda_backend.ml" "name" 1 "the backend's own name";
    e "arrayjit/lib/cuda_backend_impl.missing.ml" "name" 1 "the placeholder's name";
    e "arrayjit/lib/hip_backend.ml" "name" 1 "the backend's own name";
    e "arrayjit/lib/hip_backend.ml" "()" 1
      "reads the backend setting at inline-test init, to open the driver only when HIP is selected";
    e "arrayjit/lib/hip_backend_impl.missing.ml" "name" 1 "the placeholder's name";
    e "arrayjit/lib/metal_backend.ml" "name" 1 "the backend's own name";
    e "arrayjit/lib/metal_backend.ml" "vendor_routine_idents" 1
      "not a backend name: [metal] is a name the Metal headers declare, which a node may not take";
    e "arrayjit/lib/metal_backend_impl.missing.ml" "name" 1 "the placeholder's name";
    e "arrayjit/lib/context.ml" "cuda" 1 "the named constructor of a CUDA context";
    e "arrayjit/lib/context.ml" "hip" 1 "the named constructor of a HIP context";
    e "arrayjit/lib/context.ml" "metal" 1 "the named constructor of a Metal context";
    e "arrayjit/lib/context.ml" "cpu" 2 "the named constructor of a cc / multidev_cc context";
    e "arrayjit/lib/context.ml" "auto" 4 "the order [auto] tries the backends in, by name";
  ]

let read_sources dirs =
  List.concat_map dirs ~f:(fun dir ->
      let on_disk = "../../" ^ dir in
      let names = Stdlib.Sys.readdir on_disk |> Array.to_list in
      List.filter_map names ~f:(fun name ->
          let selected =
            Option.value_map (String.chop_suffix name ~suffix:".ml") ~default:false ~f:(fun stem ->
                List.mem names (stem ^ ".missing.ml") ~equal:String.equal)
          in
          if
            String.is_suffix name ~suffix:".ml"
            && (not (String.is_suffix name ~suffix:".pp.ml"))
            && not selected
          then Some (dir ^ "/" ^ name, Stdio.In_channel.read_all (on_disk ^ "/" ^ name))
          else None))
  |> List.sort ~compare:(fun (a, _) (b, _) -> String.compare a b)

let () =
  let vocabulary = List.map Backends.accepted_names ~f:fst in
  let sources = read_sources [ "arrayjit/lib"; "lib"; "tensor" ] in
  let sites =
    List.concat_map sources ~f:(fun (file, content) -> Literals.sites ~vocabulary ~file content)
  in
  let unlisted, stale = Literals.check ~exemptions sites in
  List.iter unlisted ~f:(fun (s : Literals.site) ->
      Stdio.eprintf "%s:%d: the backend name %S, in [%s], is in no exemption\n" s.file s.line
        s.literal s.binding);
  List.iter stale ~f:(fun (e : Literals.exemption) ->
      Stdio.eprintf
        "%s [%s]: the exemption counts %d backend-name constants, the source has another number\n"
        e.e_file e.e_binding e.e_count);
  Verdict.p "the compiler's sources are read" (List.length sources > 50);
  Verdict.p_empty "every backend-name constant in the compiler's sources is a reasoned exemption"
    ~over:sites unlisted;
  Verdict.p_empty "and every exemption counts what its binding holds" ~over:exemptions stale;
  (* The check's own controls, on synthetic sources: a decision on a name in either syntactic
     position, an alias, a constant past an exemption's count, and the boundary -- a name compared
     through a variable is not seen. *)
  let control content = Literals.sites ~vocabulary ~file:"control.ml" content in
  let unlisted content = fst (Literals.check ~exemptions:[] (control content)) in
  let refused content = not (List.is_empty (unlisted content)) in
  Verdict.p "control: a decision on a name in an expression is refused"
    (refused {|let is_gpu b = String.equal b "cuda"|});
  Verdict.p "control: a decision on a name in a pattern is refused"
    (refused {|let cap = function "hip" -> 2048 | _ -> 200|});
  Verdict.p "control: a deprecated alias is a name too"
    (refused {|let cpu b = String.equal b "sync_cc"|});
  Verdict.p "control: a constant beyond an exemption's count makes it stale"
    (let sites = control {|let name = "metal" let name = "cuda"|} in
     let _, stale =
       Literals.check
         ~exemptions:[ { e_file = "control.ml"; e_binding = "name"; e_count = 1 } ]
         sites
     in
     not (List.is_empty stale));
  let boundary = {|let is_gpu b ~gpu = String.equal b gpu|} in
  Verdict.p_empty "control: past the boundary, a name held in a variable is not seen"
    ~over:[ boundary ] (unlisted boundary)
