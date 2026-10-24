(** The reader behind [test/operations/open_node_scan] (gh-ocannl-1579): every tensor node a module
    of [tensor/] creates reaches [Tensor.note_open_node].

    A rewind discards exactly the nodes the frontend allocated since the last closure, tracked by
    uid in [Tensor.open_node_uids] (gh-ocannl-1505). A creation site that forgets the call leaves
    its node alive as a handle across a rewind; the id range the uid list replaced caught such a
    site by construction. Until this scan the pairing was documentation only.

    {1 What it reads}

    The parse tree (ppxlib's), with names resolved lexically through {!Lexical_scope}, so a
    parameter, a local binding or a local module spelled like the real thing is not the real thing.

    - {b The creators are derived, not listed.} [arrayjit/lib/tnode.ml]'s top-level values that mint
      a uid: those whose definition mentions its top-level [fresh_uid] (the one source of a node's
      identity, and the identity [note_open_node] records), closed over mentions of each other. A
      mention is any occurrence of the name resolving to that top-level binding, applied or not, so
      a wrapper or an alias of a creator is a creator. Refused: no top-level [fresh_uid]; a mention
      reaching [fresh_uid] from outside a top-level value binding with a variable pattern (a nested
      module, an extension payload, a tuple pattern, module initialization), since the scan would
      not know what to call it. Not read: the values an [include] in [tnode.ml] brings in (none
      reaches [fresh_uid] today), and any other module of [arrayjit].
    - {b The corpus} is every [.ml] under [tensor/], subdirectories included, as a source tree holds
      it: the generated lexer and parser are not sources there and are not read.
    - {b A creation} is an application of a creator through a path resolving to [Ir.Tnode]: [Ir]
      unbound by the file (the library), [Ir.Tnode], a module alias of either
      ([module Tn = Ir.Tnode]), [Tnode] after an [open Ir] (or [include Ir]), and a bare creator
      name after an [open] of [Ir.Tnode], local opens ([Tn.( … )], [let open Tn in …]) included.
      Such an open brings in exactly [tnode.ml]'s top-level value names, each shadowing a local name
      it spells. A creator referenced without being applied is refused.
    - {b Its node flows} into a [let]-bound variable when the application is in a RESULT POSITION of
      that binding's right-hand side: the expression itself, through a type constraint or coercion,
      a [let … in], [module … in] or [exception … in] body, the last expression of a sequence, every
      arm of a [match] or [if], the body and every handler of a [try], and the body of a local
      [open]. Every variable of the binding's pattern stands for its node (a creator returning a
      pair binds [let tn, init = …]; the types admit only the node to [note_open_node]). A variable
      in a result position of another binding flows on into it. Anything else -- an argument, a
      tuple or record component, a function's body, a [let*] operand, a match scrutinee -- is not
      followed, and a creation there is refused.
    - {b Noting} is an application [note_open_node x] with [x] an identifier resolving to a variable
      its node flows into, anywhere in that variable's scope. Only [tensor/tensor.ml]'s own
      top-level [note_open_node] counts, not a shadowing local one, not a same-named one in another
      file ([tensor.mli] does not export it), and a second top-level definition of it is refused.

    {1 What it deliberately does not read}

    - Control flow: a note on one path counts for the variable on every path, and a node that
      escapes before it is noted passes. The rule is "the creation's variable is noted", not a
      dominance proof.
    - [note_open_node] reached any other way: under [@@] or [|>], as a function value, through an
      alias or a wrapper. Such a node is refused as never noted -- the loud direction.
    - Modules outside [tensor/]: [lib/persistence.ml]'s load and the tile and scratch nodes the
      optimizer creates in [arrayjit/lib] are not open inference work and are not the frontend's to
      note (tensor.ml's checkpoint documentation says why). *)

open Base
open Ppxlib

type report = {
  inventory : string list;  (** Each creation the scan followed into a noted variable. *)
  findings : string list;  (** Each refusal, as [file:line: message]. *)
}

type creators = {
  names : string list;  (** The derived creators, sorted. *)
  tops : string list;  (** Every top-level value name of [tnode.ml]: what an [open] brings in. *)
  derivation_findings : string list;
}

(** The one source of a node's identity in [tnode.ml]. *)
let seed = "fresh_uid"

(** The rewind's bookkeeping lives in this file, and its top-level [note_open_node] is the one. *)
let owner_file = "tensor/tensor.ml"

let note_name = "note_open_node"

let parse ~file source =
  let lexbuf = Lexing.from_string source in
  Lexing.set_filename lexbuf file;
  Parse.implementation lexbuf

let rec pattern_var (p : pattern) =
  match p.ppat_desc with
  | Ppat_var { txt; _ } -> Some txt
  | Ppat_constraint (inner, _) -> pattern_var inner
  | _ -> None

let line (loc : Location.t) = loc.loc_start.pos_lnum

(* ---- The creators, from tnode.ml ---- *)

type top = Top of string | Local

let outside = "<outside a top-level value binding>"
let pattern_binding = "<a top-level binding with a non-variable pattern>"

let derive_creators ~file source =
  let structure = parse ~file source in
  let edges = ref [] in
  let current = ref outside in
  let final_env = ref None in
  let top_name (b : value_binding) =
    Option.value (pattern_var b.pvb_pat) ~default:pattern_binding
  in
  let walker =
    object (self)
      inherit [top, unit] Lexical_scope.scoped as super
      method local = Local
      method module_path _ _ = None

      method! ident env { txt; _ } =
        match txt with
        | Lident name -> (
            match Lexical_scope.lookup env name with
            | Some (Top defined) -> edges := (!current, defined) :: !edges
            | Some Local | None -> ())
        | Ldot _ | Lapply _ -> ()

      method! recursive_denotes ~top env bindings =
        if top then List.map bindings ~f:(fun b -> Top (top_name b))
        else super#recursive_denotes ~top env bindings

      method! define ~top env rec_flag bindings ~walk =
        if not top then super#define ~top env rec_flag bindings ~walk
        else
          let denotes = List.map bindings ~f:(fun b -> Top (top_name b)) in
          let inner =
            match rec_flag with
            | Nonrecursive -> env
            | Recursive -> self#bind_group env bindings denotes
          in
          List.iter bindings ~f:(fun b ->
              current := top_name b;
              walk inner b);
          current := outside;
          denotes

      method! finished env = final_env := Some env
    end
  in
  ignore
    (walker#structure { frames = []; modules = Map.empty (module String) } structure : structure);
  let findings = ref [] in
  (match Option.bind !final_env ~f:(fun env -> Lexical_scope.lookup env seed) with
  | Some (Top name) when String.equal name seed -> ()
  | _ ->
      findings :=
        Printf.sprintf "%s: no top-level %s binds a variable, so no creator can be derived from it"
          file seed
        :: !findings);
  let reach = Hash_set.of_list (module String) [ seed ] in
  let rec close () =
    let grew =
      List.fold !edges ~init:false ~f:(fun grew (src, dst) ->
          if Hash_set.mem reach dst && not (Hash_set.mem reach src) then (
            Hash_set.add reach src;
            true)
          else grew)
    in
    if grew then close ()
  in
  close ();
  List.iter [ outside; pattern_binding ] ~f:(fun pseudo ->
      if Hash_set.mem reach pseudo then
        findings :=
          Printf.sprintf "%s: a uid is minted %s, which the scan cannot name as a creator" file
            pseudo
          :: !findings);
  let names =
    Hash_set.to_list reach
    |> List.filter ~f:(fun n ->
        not (List.mem [ seed; outside; pattern_binding ] n ~equal:String.equal))
    |> List.sort ~compare:String.compare
  in
  if List.is_empty names then
    findings := Printf.sprintf "%s: no creator mentions %s" file seed :: !findings;
  let tops =
    List.concat_map structure ~f:(fun item ->
        match item.pstr_desc with
        | Pstr_value (_, bindings) ->
            Lexical_scope.pattern_vars (List.map bindings ~f:(fun b -> b.pvb_pat))
        | Pstr_primitive { pval_name = { txt; _ }; _ } -> [ txt ]
        | _ -> [])
    |> List.dedup_and_sort ~compare:String.compare
  in
  { names; tops; derivation_findings = List.rev !findings }

(* ---- The creation sites, in a tensor/ module ---- *)

type value = Note | Bound of int | Creator of string | Plain
type modl = Ir_lib | Tnode_mod

let loc_key (loc : Location.t) = (loc.loc_start.pos_cnum, loc.loc_end.pos_cnum)
let binding_key (b : value_binding) = b.pvb_loc.loc_start.pos_cnum

(** The result positions of an expression, as the header lists them. *)
let rec tails (e : expression) =
  match e.pexp_desc with
  | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> tails inner
  | Pexp_let (_, _, body)
  | Pexp_sequence (_, body)
  | Pexp_open (_, body)
  | Pexp_letmodule (_, _, body)
  | Pexp_letexception (_, body) ->
      tails body
  | Pexp_match (_, cases) -> List.concat_map cases ~f:(fun c -> tails c.pc_rhs)
  | Pexp_try (body, cases) -> tails body @ List.concat_map cases ~f:(fun c -> tails c.pc_rhs)
  | Pexp_ifthenelse (_, then_, else_) -> tails then_ @ Option.value_map else_ ~default:[] ~f:tails
  | _ -> [ e ]

type creation = { loc : Location.t; creator : string; target : int option; item : string }

let scan ~file ~creators source =
  let owner = String.equal file owner_file in
  let structure = parse ~file source in
  let findings = ref [] in
  let finding loc fmt = Printf.ksprintf (fun msg -> findings := (loc, msg) :: !findings) fmt in
  (* Which binding each result position belongs to, and what each binding's pattern names. *)
  let tail_of = Hashtbl.Poly.create () in
  let names_of = Hashtbl.create (module Int) in
  let collector =
    object
      inherit Ast_traverse.iter as super

      method! value_binding b =
        let key = binding_key b in
        Hashtbl.set names_of ~key ~data:(Lexical_scope.pattern_vars [ b.pvb_pat ]);
        List.iter (tails b.pvb_expr) ~f:(fun t ->
            Hashtbl.set tail_of ~key:(loc_key t.pexp_loc) ~data:key);
        super#value_binding b
    end
  in
  collector#structure structure;
  let tnode_frame =
    List.fold creators.tops
      ~init:(Map.empty (module String))
      ~f:(fun frame name ->
        Map.set frame ~key:name
          ~data:(if List.mem creators.names name ~equal:String.equal then Creator name else Plain))
  in
  let creations = ref [] in
  let flows = ref [] in
  let noted = Hash_set.create (module Int) in
  let note_definitions = ref [] in
  let item = ref "module initialization" in
  let walker =
    object (self)
      inherit [value, modl] Lexical_scope.scoped as super
      method local = Plain
      method! let_denotes b = Bound (binding_key b)

      method module_path env path =
        match path with
        | Lident name -> Map.find env.modules name
        | Ldot (prefix, "Tnode") -> (
            match self#module_path env prefix with Some Ir_lib -> Some Tnode_mod | _ -> None)
        | Ldot _ | Lapply _ -> None

      method! opened ~top:_ ~include_:_ env denotes =
        match denotes with
        | Some Ir_lib -> { env with modules = Map.set env.modules ~key:"Tnode" ~data:Tnode_mod }
        | Some Tnode_mod -> Lexical_scope.push_frame env tnode_frame
        | None -> env

      method resolve env (path : Longident.t) =
        match path with
        | Lident name -> Lexical_scope.lookup env name
        | Ldot (prefix, name) -> (
            match self#module_path env prefix with
            | Some Tnode_mod when List.mem creators.names name ~equal:String.equal ->
                Some (Creator name)
            | _ -> None)
        | Lapply _ -> None

      method! define ~top env rec_flag bindings ~walk =
        let saved = !item in
        let denotes =
          super#define ~top env rec_flag bindings ~walk:(fun inner b ->
              item := Option.value (pattern_var b.pvb_pat) ~default:saved;
              walk inner b)
        in
        item := saved;
        if top && owner then
          List.map2_exn bindings denotes ~f:(fun b denotes ->
              match pattern_var b.pvb_pat with
              | Some name when String.equal name note_name ->
                  note_definitions := b.pvb_loc :: !note_definitions;
                  Note
              | _ -> denotes)
        else denotes

      method! expression env e =
        match e.pexp_desc with
        | Pexp_apply (({ pexp_desc = Pexp_ident { txt; _ }; _ } as head), args) -> (
            match self#resolve env txt with
            | Some (Creator creator) ->
                creations :=
                  {
                    loc = e.pexp_loc;
                    creator;
                    target = Hashtbl.find tail_of (loc_key e.pexp_loc);
                    item = !item;
                  }
                  :: !creations;
                List.iter args ~f:(fun (_, arg) -> ignore (self#expression env arg : expression));
                e
            | Some Note ->
                (match args with
                | [ (Nolabel, arg) ] -> (
                    let rec strip (a : expression) =
                      match a.pexp_desc with
                      | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> strip inner
                      | _ -> a
                    in
                    match (strip arg).pexp_desc with
                    | Pexp_ident { txt = Lident name; _ } -> (
                        match Lexical_scope.lookup env name with
                        | Some (Bound key) -> Hash_set.add noted key
                        | _ -> ())
                    | _ -> ())
                | _ -> ());
                ignore (self#expression env head : expression);
                List.iter args ~f:(fun (_, arg) -> ignore (self#expression env arg : expression));
                e
            | _ -> super#expression env e)
        | Pexp_ident { txt; loc } ->
            (match self#resolve env txt with
            | Some (Creator creator) ->
                finding loc
                  "Tnode.%s is referenced without being applied: the scan follows only a creator's \
                   application into a let-bound variable"
                  creator
            | Some (Bound key) ->
                Option.iter
                  (Hashtbl.find tail_of (loc_key e.pexp_loc))
                  ~f:(fun target -> flows := (key, target) :: !flows)
            | Some Note | Some Plain | None -> ());
            super#expression env e
        | _ -> super#expression env e
    end
  in
  ignore
    (walker#structure { frames = []; modules = Map.singleton (module String) "Ir" Ir_lib } structure
      : structure);
  (match List.rev !note_definitions with
  | [] | [ _ ] -> ()
  | _ :: later ->
      List.iter later ~f:(fun loc ->
          finding loc "a second top-level %s: the scan credits one definition, not a choice"
            note_name));
  (* A variable is noted when it is passed to note_open_node, or flows into one that is. *)
  let rec close () =
    let grew =
      List.fold !flows ~init:false ~f:(fun grew (src, dst) ->
          if Hash_set.mem noted dst && not (Hash_set.mem noted src) then (
            Hash_set.add noted src;
            true)
          else grew)
    in
    if grew then close ()
  in
  close ();
  let creations =
    List.sort !creations ~compare:(fun a b ->
        Int.compare a.loc.loc_start.pos_cnum b.loc.loc_start.pos_cnum)
  in
  let names key =
    match Hashtbl.find names_of key with
    | None | Some [] -> "no variable"
    | Some names -> String.concat ~sep:", " (List.rev names)
  in
  let inventory =
    List.filter_map creations ~f:(fun c ->
        match c.target with
        | None ->
            finding c.loc
              "in %s, Tnode.%s's node is not bound by a let in a result position the scan follows, \
               so it cannot reach %s"
              c.item c.creator note_name;
            None
        | Some key when Hash_set.mem noted key ->
            Some
              (Printf.sprintf "%s: in %s, Tnode.%s -> %s: noted" file c.item c.creator (names key))
        | Some key ->
            finding c.loc "in %s, Tnode.%s's node (bound to %s) never reaches %s%s" c.item c.creator
              (names key) note_name
              (if owner then ""
               else Printf.sprintf " -- only %s's own top-level one notes it" owner_file);
            None)
  in
  let findings =
    List.sort !findings ~compare:(fun (a, _) (b, _) ->
        Int.compare a.Location.loc_start.pos_cnum b.Location.loc_start.pos_cnum)
    |> List.map ~f:(fun (loc, msg) -> Printf.sprintf "%s:%d: %s" file (line loc) msg)
  in
  { inventory; findings }
