(** A lexical scope model for the scans that resolve names in OCaml sources: which binding an
    identifier reaches at the point it is spelled, the way OCaml decides it, rather than whichever
    binding of that name the file happens to hold somewhere.

    Extracted from [Ll_test_scan] (staging#815), whose review rounds found file-wide resolution
    wrong three times over, so that [Codegen_text_scan] resolves over the same model
    (gh-ocannl-1079): there, a module alias looked up file-wide and a literal [let] keyed by its
    name alone made [codegen_text_inventory] record another binding's text for a pin, and drop the
    pins of two same-named bindings outright.

    The model is parameterised over what a name DENOTES, since each scan asks its own question of it
    -- a harness value and its class, a string literal, a module whose calls are attributed:

    - a value name resolves through [frames], innermost first: the bindings the source makes itself,
      with what each denotes, and whatever an [open]/[include] brought in, where the scan can model
      it. Every binding form scopes its names -- [let] (and [let rec] over its own right-hand
      sides), function parameters in order, match and [try] cases, [for] indices, binding operators,
      [external]s, class parameters and [let]s, an object's self and instance variables -- and a
      later structure item sees the earlier ones.
    - a module name resolves through [modules]. A later binding of the name to anything else
      replaces what it denoted -- a module, a recursive module, a functor parameter in its body, a
      [(module B)] unpack -- and a nested structure's bindings die with it, as do its opens.

    What an [open] of a module the scan cannot read brings in is not modelled: its exported names
    are unknown, so it is not taken to shadow anything. Each scan states what that costs it. *)

open Base
open Ppxlib

type ('v, 'm) env = { frames : 'v Map.M(String).t list; modules : 'm Map.M(String).t }

let rec head = function Longident.Lident s -> s | Ldot (p, _) | Lapply (p, _) -> head p

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

(** A frame of names brought in together, as an [open] of a module the scan can read brings its
    values. *)
let push_frame env frame = { env with frames = frame :: env.frames }

let lookup env name = List.find_map env.frames ~f:(fun frame -> Map.find frame name)

(** Walks a structure under {!env}. A scan instantiates it with what a name denotes ({!local}, a
    binding group's {!define}, a module path's {!module_path}) and observes the walk through
    {!ident}, called on every value identifier with the scope it is spelled in. *)
class virtual ['v, 'm] scoped =
  object (self)
    inherit [('v, 'm) env] Ast_traverse.map_with_context as super

    method virtual local : 'v
    (** What a name a pattern binds denotes: a parameter, a match case, a [for] index. *)

    method virtual module_path : ('v, 'm) env -> Longident.t -> 'm option
    (** What a module path denotes in scope, [None] for what the scan does not model. *)

    method shadowed : 'm option = None
    (** What a module name denotes once a binding the scan cannot see into -- a functor parameter,
        an unpack, a recursive module, a [struct] -- takes it over. [None] forgets the name, which
        is right where an unmodelled name means "not ours"; a scan to which an unbound name means
        the library module of that name answers a denotation of its own here instead. *)

    method ident (_ : ('v, 'm) env) (_ : longident_loc) = ()
    (** Called on every value identifier, and on a binding operator's name, with its scope. *)

    method let_denotes (_ : value_binding) : 'v = self#local
    (** What the names of one [let] binding denote, for a [let] in expression or class position. *)

    method binding_denotes (_ : ('v, 'm) env) binding = self#let_denotes binding
    (** The lexical form of {!let_denotes}, for a scan following an identifier in the RHS. *)

    method recursive_denotes ~top:(_ : bool) env bindings = self#denotations env Recursive bindings
    (** Bindings visible in recursive RHSs; defaults to the group's ordinary denotations. *)

    method define ~top:(_ : bool) env rec_flag bindings ~walk =
      let inner =
        match rec_flag with
        | Nonrecursive -> env
        | Recursive -> self#bind_group env bindings (self#denotations env rec_flag bindings)
      in
      List.iter bindings ~f:(walk inner);
      self#denotations env rec_flag bindings
    (** How a structure's binding group enters scope for the items after it: it walks the group
        itself, through [walk], and returns what each binding's names denote. [top] is whether the
        group is in the file's own structure. *)

    method declare ~top:(_ : bool) (_ : ('v, 'm) env) (_ : value_description) : 'v = self#local
    (** What an [external] declares, as {!define} for a binding group. *)

    method opened ~top:(_ : bool) ~include_:(_ : bool) env (_ : 'm option) : ('v, 'm) env = env
    (** The scope after an [open] (or, with [include_], a structure's [include]) of a module
        denoting what is given. The default brings nothing in. *)

    method item_scope ~top:(_ : bool) env (_ : structure_item) = env
    (** Scan-specific declarations entering scope after one structure item, such as constructors.
        Called after the item's own traversal and ordinary value/module/open bindings. *)

    method finished (_ : ('v, 'm) env) = ()
    (** Called with the scope at the end of the top-level structure. *)

    method module_of env module_expr =
      match module_expr.pmod_desc with
      | Pmod_constraint (inner, _) -> self#module_of env inner
      | Pmod_ident { txt; _ } -> self#module_path env txt
      | _ -> self#shadowed

    method forget env names =
      {
        env with
        modules =
          List.fold names ~init:env.modules ~f:(fun modules name ->
              match self#shadowed with
              | Some denotes -> Map.set modules ~key:name ~data:denotes
              | None -> Map.remove modules name);
      }

    method bind_module env name module_expr =
      {
        env with
        modules =
          (match self#module_of env module_expr with
          | Some denotes -> Map.set env.modules ~key:name ~data:denotes
          | None -> Map.remove env.modules name);
      }

    method bind_patterns env patterns =
      self#forget (bind_values env (pattern_vars patterns) self#local) (pattern_unpacks patterns)
    (** A pattern's value variables enter scope as {!local}, and the modules it unpacks
        ([(module B : S)]) shadow whatever those names denoted. *)

    method bind_parameters env patterns = self#bind_patterns env patterns
    (** The parameter-specific form of {!bind_patterns}, defaulting to the same local denotation. *)

    method bind_parameter env (_ : arg_label) pattern = self#bind_parameters env [ pattern ]
    (** One ordinary function parameter, with its argument label available to a scan. *)

    method bind_group env bindings denotes =
      List.fold2_exn
        (List.map bindings ~f:(fun b -> pattern_vars [ b.pvb_pat ]))
        denotes
        ~init:(self#forget env (pattern_unpacks (List.map bindings ~f:(fun b -> b.pvb_pat))))
        ~f:(fun env names denotes -> bind_values env names denotes)
    (** A binding group's names, each binding's with what {!let_denotes} gives it. *)

    method denotations env rec_flag bindings =
      let rhs_scope =
        match rec_flag with
        | Nonrecursive -> env
        | Recursive -> self#bind_patterns env (List.map bindings ~f:(fun b -> b.pvb_pat))
      in
      List.map bindings ~f:(self#binding_denotes rhs_scope)
    (** Recursive RHS names shadow the outer group before any RHS-dependent denotation is read. *)

    method bindings env rec_flag bindings =
      let denotes = self#denotations env rec_flag bindings in
      let inner =
        match rec_flag with
        | Recursive -> self#bind_group env bindings denotes
        | Nonrecursive -> env
      in
      List.iter bindings ~f:(fun b -> ignore (self#value_binding inner b : value_binding));
      self#bind_group env bindings denotes

    method! case env c =
      let inner = self#bind_patterns env [ c.pc_lhs ] in
      ignore (self#pattern env c.pc_lhs : pattern);
      Option.iter c.pc_guard ~f:(fun g -> ignore (self#expression inner g : expression));
      ignore (self#expression inner c.pc_rhs : expression);
      c

    val mutable depth = 0

    method! structure env items =
      depth <- depth + 1;
      let top = depth = 1 in
      let final =
        List.fold items ~init:env ~f:(fun env item ->
            let next =
              match item.pstr_desc with
              | Pstr_value (rec_flag, bindings) ->
                  let inner =
                    match rec_flag with
                    | Nonrecursive -> env
                    | Recursive ->
                        self#bind_group env bindings (self#recursive_denotes ~top env bindings)
                  in
                  let denotes =
                    self#define ~top env rec_flag bindings ~walk:(fun _ b ->
                        ignore (self#value_binding inner b : value_binding))
                  in
                  self#bind_group env bindings denotes
              | Pstr_primitive ({ pval_name = { txt = name; _ }; _ } as declaration) ->
                  bind_values env [ name ] (self#declare ~top env declaration)
              | Pstr_recmodule declarations ->
                  let env =
                    self#forget env (List.filter_map declarations ~f:(fun d -> d.pmb_name.txt))
                  in
                  ignore (self#structure_item env item : structure_item);
                  env
              | Pstr_module { pmb_name = { txt = Some name; _ }; pmb_expr; _ } ->
                  ignore (self#structure_item env item : structure_item);
                  self#bind_module env name pmb_expr
              | Pstr_open { popen_expr = m; _ } ->
                  ignore (self#structure_item env item : structure_item);
                  self#opened ~top ~include_:false env (self#module_of env m)
              | Pstr_include { pincl_mod = m; _ } ->
                  ignore (self#structure_item env item : structure_item);
                  self#opened ~top ~include_:true env (self#module_of env m)
              | _ ->
                  ignore (self#structure_item env item : structure_item);
                  env
            in
            self#item_scope ~top next item)
      in
      if top then self#finished final;
      depth <- depth - 1;
      items

    method! expression env e =
      match e.pexp_desc with
      | Pexp_ident ident ->
          self#ident env ident;
          super#expression env e
      | Pexp_let (rec_flag, bindings, body) ->
          ignore (self#expression (self#bindings env rec_flag bindings) body : expression);
          e
      | Pexp_function (params, constraint_, body) ->
          let env =
            List.fold params ~init:env ~f:(fun env param ->
                match param.pparam_desc with
                | Pparam_val (label, default, pattern) ->
                    Option.iter default ~f:(fun d -> ignore (self#expression env d : expression));
                    ignore (self#pattern env pattern : pattern);
                    self#bind_parameter env label pattern
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
          ignore (self#expression (self#bind_patterns env [ pattern ]) body : expression);
          e
      | Pexp_letop { let_; ands; body } ->
          let operands = let_ :: ands in
          List.iter operands ~f:(fun b ->
              self#ident env { txt = Lident b.pbop_op.txt; loc = b.pbop_op.loc };
              ignore (self#expression env b.pbop_exp : expression);
              ignore (self#pattern env b.pbop_pat : pattern));
          let inner = self#bind_patterns env (List.map operands ~f:(fun b -> b.pbop_pat)) in
          ignore (self#expression inner body : expression);
          e
      | Pexp_open ({ popen_expr; _ }, body) ->
          ignore (self#module_expr env popen_expr : module_expr);
          let inner = self#opened ~top:false ~include_:false env (self#module_of env popen_expr) in
          ignore (self#expression inner body : expression);
          e
      | Pexp_letmodule ({ txt = Some name; _ }, module_expr, body) ->
          ignore (self#module_expr env module_expr : module_expr);
          ignore (self#expression (self#bind_module env name module_expr) body : expression);
          e
      | _ -> super#expression env e

    (* A functor's parameter shadows whatever its name denoted, in the functor's body. *)
    method! module_expr env me =
      match me.pmod_desc with
      | Pmod_functor (Named ({ txt = Some name; _ }, parameter_type), body) ->
          ignore (self#module_type env parameter_type : module_type);
          ignore (self#module_expr (self#forget env [ name ]) body : module_expr);
          me
      | _ -> super#module_expr env me

    method! class_expr env ce =
      match ce.pcl_desc with
      | Pcl_fun (_, default, pattern, body) ->
          Option.iter default ~f:(fun d -> ignore (self#expression env d : expression));
          ignore (self#pattern env pattern : pattern);
          ignore (self#class_expr (self#bind_patterns env [ pattern ]) body : class_expr);
          ce
      | Pcl_let (rec_flag, bindings, body) ->
          ignore (self#class_expr (self#bindings env rec_flag bindings) body : class_expr);
          ce
      | Pcl_open ({ popen_expr = { txt = path; _ }; _ }, body) ->
          let inner = self#opened ~top:false ~include_:false env (self#module_path env path) in
          ignore (self#class_expr inner body : class_expr);
          ce
      | _ -> super#class_expr env ce

    (* An object's self, instance variables and ancestor names ([inherit c as a]) are in scope in
       every field -- the whole body, which errs toward the source's own. *)
    method! class_structure env cs =
      ignore (self#pattern env cs.pcstr_self : pattern);
      let vals =
        List.filter_map cs.pcstr_fields ~f:(fun field ->
            match field.pcf_desc with
            | Pcf_val ({ txt; _ }, _, _) | Pcf_inherit (_, _, Some { txt; _ }) -> Some txt
            | _ -> None)
      in
      let inner = bind_values (self#bind_patterns env [ cs.pcstr_self ]) vals self#local in
      List.iter cs.pcstr_fields ~f:(fun field ->
          ignore (self#class_field inner field : class_field));
      cs
  end
