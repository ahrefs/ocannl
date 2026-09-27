open Base
(** A deliberately syntactic ratchet, not a type checker. Constructor names are derived from
    Low_level's t/scalar_t declarations, independent of qualifier spelling (including opens and
    aliases). A same-named foreign constructor is conservatively counted. Expressions and patterns
    are separate: one record construction or one recursive binding inspecting at least two distinct
    IR constructors warrants the harness. Quoted fixtures and comments are not AST nodes. Adoption
    is linking the harness AND calling into its IR surface ({!surface}); linking it for an operand
    helper alone adopts nothing (gh-ocannl-1052). This does not ban test-specific match arms. *)

open Ppxlib
module Dune = Dune_stanza_scan
module Text = Codegen_text_scan

type census = { records : int; traversals : int }
type exemption = Migration of census | Permanent

(** How far a test with detected debt has come: its stanzas do not link the harness, they link it
    but the source calls nothing of its IR surface (the gh-ocannl-1052 state: linked for
    [Ll_test.cycle] alone), or it links the harness and uses it. Only the last is adoption. *)
type adoption = Unlinked | Linked_unused | Adopted

(** The harness, as each module's name and its source relative to the workspace root, in dependency
    order ([Ll_test] includes [Ll_builders]). The names are also how a test reaches the harness's
    values — a qualifier, alias, [open] or [include] of one — beside the libraries {!links_harness}
    reads. *)
let harness_sources =
  [ ("Ll_builders", "test/support/ll_builders.ml"); ("Ll_test", "test/support/ll_test.ml") ]

type harness_module = { values : bool Map.M(String).t; ir_aliases : string list }
(** One harness module as classified: every value it exports mapped to whether it belongs to the IR
    surface, and the module aliases of [Ir] it exports ([LL] in [Ll_builders]), which an [include]
    or [open] of it brings into scope. Kept per module because the two can disagree on a name:
    [Ll_test] includes [Ll_builders] and may redefine what it included, and the later definition is
    the one [Ll_test.name] calls. *)

type surface = harness_module Map.M(String).t

(** [members surface module_name ~ir] is the sorted names [module_name] exports inside ([~ir:true])
    or outside the IR surface. *)
let members (surface : surface) module_name ~ir =
  Option.value_map (Map.find surface module_name) ~default:[] ~f:(fun m ->
      Map.keys (Map.filter m.values ~f:(Bool.equal ir)))

let is_ir (surface : surface) module_name name =
  Option.value ~default:false
    (Option.bind (Map.find surface module_name) ~f:(fun m -> Map.find m.values name))

let rec head = function Longident.Lident s -> s | Ldot (p, _) | Lapply (p, _) -> head p

(** {2 Lexical scope}

    Both halves of the adoption check — classifying the harness's own values and crediting a test's
    calls — resolve names the way OCaml does, over one scope model, so the class a use is judged by
    is the class of the binding it actually reaches:

    - a value name resolves through [frames], innermost first: a binding the source makes itself
      ([Local], or the harness's own top-level definitions with their class), or the values of a
      harness module an [open]/[include] brought in. Every binding form scopes its names — [let]
      (and [let rec] over its own right-hand sides), function parameters in order, match and [try]
      cases, [for] indices, binding operators, [external]s, class parameters and [let]s, an object's
      self and instance variables — and a later structure item sees the earlier ones.
    - a module name resolves through [modules]: the harness modules by their exact names, aliases of
      them ([module L = Ll_test], [let module L = Ll_test in], through a signature constraint), and
      aliases of [Ir]. A later binding of the name to anything else removes it — a module, a
      recursive module, a functor parameter in its body, a [(module B)] unpack — and a nested
      structure's bindings die with it. Only an exact path counts: [Outer.Ll_test.seq] is a module
      named [Ll_test] inside [Outer], not the harness.

    One boundary is deliberate. An [open] of a module the scan cannot read ([Base],
    [Verdict.Claims]) is opaque: its exported names are not modeled, so a later such open is not
    taken to shadow a harness name opened before it. Taking it to would refuse every adopted test in
    the tree, which writes [open Ll_test] then [open Verdict.Claims]; shadowing a builder needs a
    module exporting a value of the builder's name, which no module those tests open does. *)

type denotes = Local | Harness of bool
type module_denotes = Harness_module of string | Ir_root
type env = { frames : denotes Map.M(String).t list; modules : module_denotes Map.M(String).t }

let pattern_vars patterns =
  let found = ref [] in
  let collector =
    object
      inherit Ast_traverse.iter as super

      method! pattern p =
        (match p.ppat_desc with
        | Ppat_var { txt; _ } | Ppat_alias (_, { txt; _ }) -> found := txt :: !found
        | _ -> ());
        super#pattern p
    end
  in
  List.iter patterns ~f:collector#pattern;
  !found

let bind_values env names denotes =
  match names with
  | [] -> env
  | _ ->
      let frame =
        Map.of_alist_reduce
          (module String)
          (List.map names ~f:(fun name -> (name, denotes)))
          ~f:(fun _ later -> later)
      in
      { env with frames = frame :: env.frames }

let pattern_unpacks patterns =
  let found = ref [] in
  let collector =
    object
      inherit Ast_traverse.iter as super

      method! pattern p =
        (match p.ppat_desc with
        | Ppat_unpack { txt = Some name; _ } -> found := name :: !found
        | _ -> ());
        super#pattern p
    end
  in
  List.iter patterns ~f:collector#pattern;
  !found

let forget_modules env names =
  { env with modules = List.fold names ~init:env.modules ~f:Map.remove }

(** A pattern's value variables enter scope as the source's own, and the modules it unpacks
    ([(module B : S)]) shadow whatever those names denoted. *)
let bind_patterns env patterns =
  forget_modules (bind_values env (pattern_vars patterns) Local) (pattern_unpacks patterns)

let lookup env name = List.find_map env.frames ~f:(fun frame -> Map.find frame name)

let rec module_of env module_expr =
  match module_expr.pmod_desc with
  | Pmod_constraint (inner, _) -> module_of env inner
  | Pmod_ident { txt = Lident name; _ } -> Map.find env.modules name
  | Pmod_ident { txt; _ } -> (
      match Map.find env.modules (head txt) with Some Ir_root -> Some Ir_root | _ -> None)
  | _ -> None

let bind_module env name module_expr =
  {
    env with
    modules =
      (match module_of env module_expr with
      | Some denotes -> Map.set env.modules ~key:name ~data:denotes
      | None -> Map.remove env.modules name);
  }

let rec open_module (surface : surface) env module_expr =
  open_denoted surface env (module_of env module_expr)

and open_denoted (surface : surface) env denoted =
  match denoted with
  | Some (Harness_module h) ->
      let m = Map.find_exn surface h in
      {
        frames = Map.map m.values ~f:(fun ir -> Harness ir) :: env.frames;
        modules =
          List.fold m.ir_aliases ~init:env.modules ~f:(fun modules alias ->
              Map.set modules ~key:alias ~data:Ir_root);
      }
  | Some Ir_root | None -> env

(** What a value identifier reaches: [Some (Harness ir)] for a harness value, [Some Local] for the
    source's own binding, [None] for anything the scope model does not know. *)
let resolve (surface : surface) env = function
  | Longident.Lident name -> lookup env name
  | Ldot (Lident qualifier, name) -> (
      match Map.find env.modules qualifier with
      | Some (Harness_module h) -> Some (Harness (is_ir surface h name))
      | _ -> None)
  | _ -> None

(** Walks a structure under {!env}, calling [use] on every value identifier with what it resolves to
    and [mentions_ir] on every path rooted at an [Ir] alias. [define] decides how a top-level
    binding group enters scope for the items after it: it walks the group itself, through [walk],
    and returns what each of its names denotes. *)
class virtual scoped (surface : surface) =
  object (self)
    inherit [env] Ast_traverse.map_with_context as super
    method virtual use : denotes option -> unit
    method virtual mentions_ir : unit

    method virtual define
        : top:bool ->
          env ->
          rec_flag ->
          value_binding list ->
          walk:(env -> value_binding -> unit) ->
          denotes list

    method virtual declare : top:bool -> env -> value_description -> denotes
    (** What an [external] declares, as {!define} for a binding group. *)

    method included (_ : string) = ()
    (** Called for each top-level [include] of a harness module, in order with {!define}. *)

    method finished (_ : env) = ()
    (** Called with the scope at the end of the top-level structure. *)

    val mutable depth = 0

    method! longident env l =
      (match Map.find env.modules (head l) with Some Ir_root -> self#mentions_ir | _ -> ());
      l

    method bindings env rec_flag bindings =
      let patterns = List.map bindings ~f:(fun b -> b.pvb_pat) in
      let inner =
        match rec_flag with Recursive -> bind_patterns env patterns | Nonrecursive -> env
      in
      List.iter bindings ~f:(fun b -> ignore (self#value_binding inner b : value_binding));
      bind_patterns env patterns

    method! case env c =
      let inner = bind_patterns env [ c.pc_lhs ] in
      ignore (self#pattern env c.pc_lhs : pattern);
      Option.iter c.pc_guard ~f:(fun g -> ignore (self#expression inner g : expression));
      ignore (self#expression inner c.pc_rhs : expression);
      c

    method! structure env items =
      depth <- depth + 1;
      let top = depth = 1 in
      let final =
        List.fold items ~init:env ~f:(fun env item ->
            match item.pstr_desc with
            | Pstr_value (rec_flag, bindings) ->
                let patterns = List.map bindings ~f:(fun b -> b.pvb_pat) in
                let inner =
                  match rec_flag with
                  | Recursive -> bind_patterns env patterns
                  | Nonrecursive -> env
                in
                let classes =
                  self#define ~top env rec_flag bindings ~walk:(fun _ b ->
                      ignore (self#value_binding inner b : value_binding))
                in
                List.fold2_exn
                  (List.map bindings ~f:(fun b -> pattern_vars [ b.pvb_pat ]))
                  classes
                  ~init:(forget_modules env (pattern_unpacks patterns))
                  ~f:(fun env names denotes -> bind_values env names denotes)
            | Pstr_primitive ({ pval_name = { txt = name; _ }; _ } as declaration) ->
                bind_values env [ name ] (self#declare ~top env declaration)
            | Pstr_recmodule declarations ->
                let env =
                  forget_modules env (List.filter_map declarations ~f:(fun d -> d.pmb_name.txt))
                in
                ignore (self#structure_item env item : structure_item);
                env
            | Pstr_module { pmb_name = { txt = Some name; _ }; pmb_expr; _ } ->
                ignore (self#structure_item env item : structure_item);
                bind_module env name pmb_expr
            | Pstr_open { popen_expr = m; _ } | Pstr_include { pincl_mod = m; _ } ->
                ignore (self#structure_item env item : structure_item);
                (match (item.pstr_desc, module_of env m) with
                | Pstr_include _, Some (Harness_module h) when top -> self#included h
                | _ -> ());
                open_module surface env m
            | _ ->
                ignore (self#structure_item env item : structure_item);
                env)
      in
      if top then self#finished final;
      depth <- depth - 1;
      items

    method! expression env e =
      match e.pexp_desc with
      | Pexp_ident { txt; _ } ->
          self#use (resolve surface env txt);
          super#expression env e
      | Pexp_let (rec_flag, bindings, body) ->
          ignore (self#expression (self#bindings env rec_flag bindings) body : expression);
          e
      | Pexp_function (params, constraint_, body) ->
          let env =
            List.fold params ~init:env ~f:(fun env param ->
                match param.pparam_desc with
                | Pparam_val (_, default, pattern) ->
                    Option.iter default ~f:(fun d -> ignore (self#expression env d : expression));
                    ignore (self#pattern env pattern : pattern);
                    bind_patterns env [ pattern ]
                | Pparam_newtype _ -> env)
          in
          Option.iter constraint_ ~f:(fun c ->
              ignore (self#type_constraint env c : type_constraint));
          (match body with
          | Pfunction_body b -> ignore (self#expression env b : expression)
          | Pfunction_cases (cases, _, _) ->
              List.iter cases ~f:(fun c -> ignore (self#case env c : case)));
          e
      | Pexp_match (scrutinee, cases) | Pexp_try (scrutinee, cases) ->
          ignore (self#expression env scrutinee : expression);
          List.iter cases ~f:(fun c -> ignore (self#case env c : case));
          e
      | Pexp_for (pattern, low, high, _, body) ->
          ignore (self#expression env low : expression);
          ignore (self#expression env high : expression);
          ignore (self#expression (bind_patterns env [ pattern ]) body : expression);
          e
      | Pexp_letop { let_; ands; body } ->
          let operands = let_ :: ands in
          List.iter operands ~f:(fun b -> ignore (self#expression env b.pbop_exp : expression));
          let inner = bind_patterns env (List.map operands ~f:(fun b -> b.pbop_pat)) in
          ignore (self#expression inner body : expression);
          e
      | Pexp_open ({ popen_expr; _ }, body) ->
          ignore (self#module_expr env popen_expr : module_expr);
          ignore (self#expression (open_module surface env popen_expr) body : expression);
          e
      | Pexp_letmodule ({ txt = Some name; _ }, module_expr, body) ->
          ignore (self#module_expr env module_expr : module_expr);
          ignore (self#expression (bind_module env name module_expr) body : expression);
          e
      | _ -> super#expression env e

    (* A functor's parameter shadows whatever its name denoted, in the functor's body. *)
    method! module_expr env me =
      match me.pmod_desc with
      | Pmod_functor (Named ({ txt = Some name; _ }, parameter_type), body) ->
          ignore (self#module_type env parameter_type : module_type);
          ignore (self#module_expr (forget_modules env [ name ]) body : module_expr);
          me
      | _ -> super#module_expr env me

    method! class_expr env ce =
      match ce.pcl_desc with
      | Pcl_fun (_, default, pattern, body) ->
          Option.iter default ~f:(fun d -> ignore (self#expression env d : expression));
          ignore (self#class_expr (bind_patterns env [ pattern ]) body : class_expr);
          ce
      | Pcl_let (rec_flag, bindings, body) ->
          ignore (self#class_expr (self#bindings env rec_flag bindings) body : class_expr);
          ce
      | Pcl_open ({ popen_expr = { txt = path; _ }; _ }, body) ->
          let denoted = match path with Lident name -> Map.find env.modules name | _ -> None in
          ignore (self#class_expr (open_denoted surface env denoted) body : class_expr);
          ce
      | _ -> super#class_expr env ce

    (* An object's self and its instance variables are in scope in every field. *)
    method! class_structure env cs =
      let vals =
        List.filter_map cs.pcstr_fields ~f:(fun field ->
            match field.pcf_desc with Pcf_val ({ txt; _ }, _, _) -> Some txt | _ -> None)
      in
      let inner = bind_values (bind_patterns env [ cs.pcstr_self ]) vals Local in
      List.iter cs.pcstr_fields ~f:(fun field ->
          ignore (self#class_field inner field : class_field));
      cs
  end

let root_env (surface : surface) =
  {
    frames = [];
    modules =
      Map.of_alist_exn
        (module String)
        (("Ir", Ir_root) :: List.map (Map.keys surface) ~f:(fun h -> (h, Harness_module h)));
  }

(** [surface sources] classifies each harness module's top-level values, DERIVED from their
    definitions rather than listed, so the next helper added to [Ll_test] lands in the right class
    without anyone deciding it: a value is IR when its definition mentions a path rooted at [Ir] or
    an alias of it ([module LL = Ir.Low_level], its own or one an [include] brought in) — body, type
    annotation, record field — or calls a value that resolves, in {!env}, to an IR value of a
    harness module or of this one. That puts every builder, traversal and pipeline helper in the
    surface ([tick] only through [add]/[c]/[embed]), and leaves the operand-data helpers ([cycle],
    [cycle_flat], [weighted], [drift], [flat]) and the value checks ([blank], [close], [same])
    outside: integer and float arithmetic with no IR in it. Each binding of a group is classified on
    its own; a [let rec] group, whose members can reach each other, together; a pattern binding
    several names credits none of them. An [external] is classified by its declared type. *)
let surface sources : surface =
  List.fold sources
    ~init:(Map.empty (module String))
    ~f:(fun (classified : surface) (module_name, source) ->
      let reaches = ref false in
      let exported = ref (Map.empty (module String)) and ir_aliases = ref [] in
      let classifier =
        object (self)
          inherit scoped classified
          method use = function Some (Harness true) -> reaches := true | _ -> ()
          method mentions_ir = reaches := true

          (* A nested definition's evidence also counts for the top-level binding containing it. *)
          method define ~top env rec_flag bindings ~walk =
            let classify group =
              let outer = !reaches in
              reaches := false;
              List.iter group ~f:(walk env);
              let ir = !reaches in
              reaches := outer || ir;
              ir
            in
            let classes =
              match rec_flag with
              | Recursive ->
                  let ir = classify bindings in
                  List.map bindings ~f:(fun _ -> ir)
              | Nonrecursive -> List.map bindings ~f:(fun b -> classify [ b ])
            in
            (* A pattern binding several names ([let a, b = ...]) gets one piece of evidence for all
               of them, so none is credited: the class would be a guess for each component. *)
            let classes =
              List.map2_exn bindings classes ~f:(fun b ir ->
                  ir && List.length (pattern_vars [ b.pvb_pat ]) <= 1)
            in
            if top then
              List.iter2_exn bindings classes ~f:(fun b ir ->
                  List.iter (pattern_vars [ b.pvb_pat ]) ~f:(fun name ->
                      exported := Map.set !exported ~key:name ~data:ir));
            List.map classes ~f:(fun ir -> Harness ir)

          method declare ~top env declaration =
            let outer = !reaches in
            reaches := false;
            ignore (self#value_description env declaration : value_description);
            let ir = !reaches in
            reaches := outer || ir;
            if top then exported := Map.set !exported ~key:declaration.pval_name.txt ~data:ir;
            Harness ir

          (* An include re-exports what it brings in, over what was defined before it. *)
          method! included h =
            exported :=
              Map.merge_skewed !exported (Map.find_exn classified h).values
                ~combine:(fun ~key:_ _ included -> included)

          method! finished env =
            ir_aliases :=
              Map.fold env.modules ~init:[] ~f:(fun ~key ~data acc ->
                  match data with
                  | Ir_root when not (String.equal key "Ir") -> key :: acc
                  | _ -> acc)
        end
      in
      ignore (classifier#structure (root_env classified) (Text.structure_of source) : structure);
      Map.set classified ~key:module_name ~data:{ values = !exported; ir_aliases = !ir_aliases })

(** [uses_surface ~surface source]: whether the test calls an IR value of the harness — a call that
    resolves, in {!env}, to one: qualified by a harness module or an alias of one ([Ll_test.set],
    [module B = Ll_builders] then [B.loop_n]), or unqualified under an [open]/[include] of one, by
    the class that module gives the name. A name the test binds for itself, where that binding is in
    scope, is its own and not the builder's. *)
let uses_surface ~(surface : surface) source =
  let found = ref false in
  let consumer =
    object
      inherit scoped surface
      method use = function Some (Harness true) -> found := true | _ -> ()
      method mentions_ir = ()

      method define ~top:_ env _ bindings ~walk =
        List.iter bindings ~f:(walk env);
        List.map bindings ~f:(fun _ -> Local)

      method declare ~top:_ _ _ = Local
    end
  in
  ignore (consumer#structure (root_env surface) (Text.structure_of source) : structure);
  !found

let constructors source =
  let names = ref [] and records = ref [] in
  Parse.implementation (Lexing.from_string source)
  |> List.iter ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_type (_, declarations) ->
          List.iter declarations ~f:(fun declaration ->
              if List.mem [ "t"; "scalar_t" ] declaration.ptype_name.txt ~equal:String.equal then
                match declaration.ptype_kind with
                | Ptype_variant ctors ->
                    List.iter ctors ~f:(fun c ->
                        names := c.pcd_name.txt :: !names;
                        match c.pcd_args with
                        | Pcstr_record _ -> records := c.pcd_name.txt :: !records
                        | _ -> ())
                | _ -> ())
      | _ -> ());
  (Set.of_list (module String) !names, Set.of_list (module String) !records)

let census ~constructors:(constructors, record_constructors) source =
  let known ident = Set.mem constructors (Longident.last_exn ident) in
  let records = ref 0 and traversals = ref 0 in
  let inspect bindings =
    List.iter bindings ~f:(fun binding ->
        let patterns = Hash_set.create (module String) in
        let walker =
          object (self)
            inherit Ast_traverse.iter as super

            method! structure_item item =
              match item.pstr_desc with
              | Pstr_value (Recursive, _) -> ()
              | _ -> super#structure_item item

            method! expression expr =
              match expr.pexp_desc with
              | Pexp_let (Recursive, _, body) -> self#expression body
              | _ -> super#expression expr

            method! pattern p =
              (match p.ppat_desc with
              | Ppat_construct ({ txt; _ }, _) when known txt ->
                  Hash_set.add patterns (Longident.last_exn txt)
              | _ -> ());
              (* Nested subpatterns matter; nested value definitions have their own census. *)
              super#pattern p
          end
        in
        walker#expression binding.pvb_expr;
        if Hash_set.length patterns >= 2 then Int.incr traversals)
  in
  let walker =
    object
      inherit Ast_traverse.iter as super

      method! structure_item item =
        (match item.pstr_desc with Pstr_value (Recursive, bindings) -> inspect bindings | _ -> ());
        super#structure_item item

      method! expression expr =
        (match expr.pexp_desc with
        | Pexp_construct ({ txt; _ }, Some { pexp_desc = Pexp_record _; _ })
          when Set.mem record_constructors (Longident.last_exn txt) ->
            Int.incr records
        | Pexp_let (Recursive, bindings, _) -> inspect bindings
        | _ -> ());
        super#expression expr
    end
  in
  walker#structure (Parse.implementation (Lexing.from_string source));
  { records = !records; traversals = !traversals }

let needs_harness { records; traversals } = records >= 1 || traversals >= 1

let stanza_owns ~directory_modules ~module_name ~stanzas stanza =
  Option.value_map (Dune.head stanza) ~default:false ~f:(fun head ->
      List.mem Dune.module_bearing_heads head ~equal:String.equal)
  && List.exists (Dune.modules_of ~directory_modules stanzas stanza) ~f:(fun name ->
      String.Caseless.equal name module_name)

let links_harness stanza =
  Option.value_map (Dune.field stanza "libraries") ~default:false ~f:(fun libraries ->
      List.exists [ "ll_test"; "arrayjit.ll_builders" ] ~f:(fun name ->
          List.mem libraries (Sexp.Atom name) ~equal:Sexp.equal))

let stanza_links ~directory_modules ~module_name ~stanzas stanza =
  stanza_owns ~directory_modules ~module_name ~stanzas stanza && links_harness stanza

let linked ~directory_modules ~module_name stanzas =
  List.exists stanzas ~f:(stanza_links ~directory_modules ~module_name ~stanzas)

(** A select arm is source for the generated target module, never a module named *.real or
    *.missing. Keep the owning stanza with the relationship: another stanza linking ll_test cannot
    cover it, and every owner of a reused arm must adopt the harness. *)
let select_arms stanza =
  Option.value (Dune.field stanza "libraries") ~default:[]
  |> List.concat_map ~f:(function
    | Sexp.List (Sexp.Atom "select" :: Sexp.Atom target :: Sexp.Atom "from" :: arms)
      when String.is_suffix target ~suffix:".ml" ->
        List.filter_map arms ~f:(function
          | Sexp.List terms -> (
              match
                List.drop_while terms ~f:(fun term -> not (Sexp.equal term (Sexp.Atom "->")))
              with
              | [ Sexp.Atom "->"; Sexp.Atom source ] -> Some (target, source)
              | _ -> None)
          | _ -> None)
    | _ -> [])

(** Fold physical dune files and parent subdir blocks into the same directory groups before
    resolving defaults. Directory-spanning ownership modes remain explicit refusals, as in
    env_var_deps; silently treating them as unlinked would misdiagnose an adopted test. *)
let test_source path =
  (String.is_prefix path ~prefix:"test/" || String.is_prefix path ~prefix:"arrayjit/test/")
  && String.is_suffix path ~suffix:".ml"
  && not
       (String.equal (Stdlib.Filename.dirname path) "test/ppx"
       && String.is_suffix path ~suffix:"_expected.ml")

(** Literal copy forms used by this tree. Do not silently guess at globs, pforms or generated inputs
    outside the declared test corpus. Non-ML copies do not affect module ownership. *)
let copy_input stanza =
  match Dune.head stanza with
  | Some ("copy_files" | "copy_files#") -> (
      let input =
        match stanza with
        | Sexp.List [ _; Sexp.Atom input ] -> Some input
        | _ -> (
            match Dune.field stanza "files" with
            | Some [ Sexp.Atom input ] -> Some input
            | _ -> None)
      in
      match input with
      | None -> Error "unsupported copy_files source form"
      | Some input ->
          let dynamic s = String.exists s ~f:(fun c -> String.mem "*?[]{}%" c) in
          let extension = Stdlib.Filename.extension input in
          if
            (not (String.is_empty extension))
            && (not (dynamic extension))
            && not (String.equal extension ".ml")
          then Ok None
          else if dynamic input then Error "unsupported copy_files glob or dynamic source"
          else Ok (Option.some_if (String.is_suffix input ~suffix:".ml") input))
  | _ -> Ok None

let ownership ~sources ~dune_files =
  let groups =
    List.concat_map dune_files ~f:(fun (path, content) ->
        let dir = Stdlib.Filename.dirname path in
        Dune.walk dir (Dune.stanzas content) ~f:(fun dir stanza ->
            let dir = Dune.normalize_path dir in
            [ ((if String.is_empty dir then "." else dir), stanza) ]))
    |> Map.of_alist_multi (module String)
  in
  let problems =
    Map.to_alist groups
    |> List.concat_map ~f:(fun (dir, stanzas) ->
        if
          List.exists sources ~f:(fun path ->
              String.equal dir "." || String.is_prefix path ~prefix:(dir ^ "/"))
        then
          List.filter_map stanzas ~f:(fun stanza ->
              match stanza with
              | Sexp.List [ Sexp.Atom "include_subdirs"; Sexp.Atom "no" ] -> None
              | Sexp.List (Sexp.Atom (("include_subdirs" | "include") as directive) :: _) ->
                  Some (dir ^ ": unsupported module ownership directive " ^ directive)
              | _ -> None)
        else [])
  in
  let selections =
    Map.to_alist groups
    |> List.concat_map ~f:(fun (dir, stanzas) ->
        List.concat_map stanzas ~f:(fun stanza ->
            select_arms stanza
            |> List.map ~f:(fun (target, arm) ->
                (dir, stanza, target, Dune.normalize_path (Dune.in_subdir dir arm)))))
  in
  let copy_errors = ref [] in
  let copies =
    Map.to_alist groups
    |> List.concat_map ~f:(fun (dir, stanzas) ->
        List.filter_map stanzas ~f:(fun stanza ->
            match copy_input stanza with
            | Error reason ->
                if test_source (Dune.normalize_path (Dune.in_subdir dir "probe.ml")) then
                  copy_errors := (dir ^ ": " ^ reason) :: !copy_errors;
                None
            | Ok None -> None
            | Ok (Some input) ->
                Some
                  ( Dune.normalize_path (Dune.in_subdir dir input),
                    Dune.normalize_path (Dune.in_subdir dir (Stdlib.Filename.basename input)) )))
  in
  let inputs =
    copies
    @ List.map selections ~f:(fun (dir, _, target, arm) ->
        (arm, Dune.normalize_path (Dune.in_subdir dir target)))
  in
  let rec declared_origin seen path =
    if List.mem sources path ~equal:String.equal then true
    else if List.mem seen path ~equal:String.equal then false
    else
      let origins =
        List.filter_map inputs ~f:(fun (input, target) ->
            Option.some_if (String.equal target path) input)
      in
      (not (List.is_empty origins)) && List.for_all origins ~f:(declared_origin (path :: seen))
  in
  let input_errors =
    List.filter_map inputs ~f:(fun (input, target) ->
        if test_source target && not (declared_origin [] input) then
          Some (target ^ ": source input outside declared test corpus: " ^ input)
        else None)
  in
  let module_name path = Stdlib.Filename.remove_extension (Stdlib.Filename.basename path) in
  let directory_modules dir =
    List.filter_map
      (sources @ List.map copies ~f:snd)
      ~f:(fun source ->
        if
          String.equal (Stdlib.Filename.dirname source) dir
          && not (List.exists selections ~f:(fun (_, _, _, arm) -> String.equal source arm))
        then Some (module_name source)
        else None)
    @ List.filter_map selections ~f:(fun (owner_dir, _, target, _) ->
        Option.some_if (String.equal owner_dir dir) (module_name target))
    |> List.dedup_and_sort ~compare:String.compare
  in
  let stanzas_at dir = Option.value (Map.find groups dir) ~default:[] in
  let rec owners seen path =
    if List.mem seen path ~equal:String.equal then [ false ]
    else
      let seen = path :: seen in
      let direct =
        match List.filter selections ~f:(fun (_, _, _, arm) -> String.equal path arm) with
        | [] ->
            if not (test_source path) then []
            else
              let dir = Stdlib.Filename.dirname path in
              List.filter (stanzas_at dir)
                ~f:
                  (stanza_owns ~directory_modules:(directory_modules dir)
                     ~module_name:(module_name path) ~stanzas:(stanzas_at dir))
              |> List.map ~f:links_harness
        | selected ->
            List.concat_map selected ~f:(fun (dir, stanza, target, _) ->
                stanza_links ~directory_modules:(directory_modules dir)
                  ~module_name:(module_name target) ~stanzas:(stanzas_at dir) stanza
                :: copied_owners seen (Dune.normalize_path (Dune.in_subdir dir target)))
      in
      direct @ copied_owners seen path
  and copied_owners seen path =
    List.filter copies ~f:(fun (source, _) -> String.equal source path)
    |> List.concat_map ~f:(fun (_, target) -> owners seen target)
  in
  let is_linked path =
    match owners [] path with [] -> false | owners -> List.for_all owners ~f:Fn.id
  in
  (is_linked, problems @ List.rev !copy_errors @ input_errors)

let violations ~exemptions rows =
  let debt =
    List.filter rows ~f:(fun (_, counts, adoption) ->
        needs_harness counts && match adoption with Adopted -> false | _ -> true)
  in
  let missing =
    List.filter_map debt ~f:(fun (path, counts, adoption) ->
        match List.find exemptions ~f:(fun (name, _, _) -> String.equal name path) with
        | None -> (
            match adoption with
            | Linked_unused ->
                Some
                  (path
                 ^ ": links ll_test but calls none of its IR surface, so its hand-built Low_level \
                    is still debt; use the harness's builders or name a migration exemption")
            | Unlinked | Adopted -> Some (path ^ ": requires ll_test or a named migration exemption")
            )
        | Some (_, Permanent, _) -> None
        | Some (_, Migration cap, _) ->
            if counts.records <= cap.records && counts.traversals <= cap.traversals then None
            else
              Some
                (Printf.sprintf
                   "%s: migration debt grew beyond ll_test baseline (records %d/%d, traversals \
                    %d/%d)"
                   path counts.records cap.records counts.traversals cap.traversals))
  in
  let stale =
    List.filter_map exemptions ~f:(fun (path, _, _) ->
        if List.exists debt ~f:(fun (name, _, _) -> String.equal name path) then None
        else Some (path ^ ": stale ll_test exemption"))
  in
  missing @ stale
