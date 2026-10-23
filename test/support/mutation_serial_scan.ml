(** The reader behind [test/operations/mutation_serial_scan] (gh-ocannl-1506): every write to a
    module-level mutable binding of the inference-session modules ([tensor/row.ml],
    [tensor/shape.ml]) advances [Row.mutation_serial], or is classified in place as not needing to.

    The serial is how a failed construction is told apart from one that failed before touching the
    session (gh-ocannl-1301): a write it does not count is a construction that fails after mutating
    the session without tainting it. Checkpoint rollback (gh-ocannl-903) leans on the same
    obligation for the session's module-level state. Until this scan it was documentation only.

    {1 What it reads}

    The parse tree (ppxlib's), with names resolved lexically through {!Lexical_scope}, so a
    parameter or local binding spelled like a table is not the table.

    - {b Tracked binding}: a [let x = e] of the file's own top-level structure whose [e], under type
      constraints, is an allocation: [ref _] (also [Stdlib.ref], [Base.ref]), or [M.create _] /
      [M.make _] for [M] one of [Hashtbl], [Hashtbl.Poly], [Hash_set], [Hash_set.Poly], [Queue],
      [Stack], [Array].
    - {b Occurrences} of a tracked name are classified by the application they are the first
      positional argument of, from closed lists ({!ref_reads}, {!ref_writes}, {!family_functions}):
      [!x] reads a ref; [x := _], [incr x], [decr x], [Int.incr x], [Int.decr x] write it; a
      container is read or written by the listed functions of its own family ([Hashtbl.find_or_add]
      is a write, [Hash_set.mem] a read; [a.(i) <- v] is [Array.set]). A [Stdlib.] or [Base.]
      qualifier is dropped first.
    - {b Accounting}: a write is accounted for when its nearest enclosing let-bound function (any
      [let f … = fun …] at any depth; the top-level definition when none encloses it) contains a
      call [note_mutation ()] anywhere in its body, nested closures included -- so the [~default]
      closure of a [find_or_add] counts. The call must reach the real one: [Row.note_mutation] with
      no module named [Row] bound in scope (a local module, functor parameter or unpack), or in
      [row.ml] the top-level [note_mutation] itself, not a shadowing local. An [open] that brings a
      [Row] in is not modelled.
    - {b In-place classification}, as attributes with a closed grammar; a reason is a string literal
      of more than one word:
    - [[@@mutation_serial.exempt "<reason>"]] on a tracked binding: its writes need no note
      (fresh-id counters, the serial itself, a solver pass's transient flags). Also the only way to
      admit state the scan cannot read (below).
    - [[@@mutation_serial.fingerprinted]] on a tracked [ref] that [Shape.fingerprint] compares
      physically: held to the relationship, not to a reason -- the file's top-level [mutated_since]
      must read [!x].
    - [[@@mutation_serial.boundary "<reason>"]] on a let-bound function that is a session boundary
      (the commit, a rewind, a reset): its own writes need no note.

    {1 What it refuses}

    Each of these is a finding, never a silent pass:

    - a write not accounted for; a write at module initialization, under no function;
    - any other use of a tracked name: passed to a helper, aliased, a container function outside the
      lists, a ref given to a container function;
    - an allocation at a binding head outside the list above ([Foo.create], [Foo.make] for any other
      [Foo]) -- unread, so name it or extend the list;
    - module-level state the scan does not resolve: an allocation bound by a non-variable pattern,
      one evaluated at module initialization but hidden under a local [let] of a top-level value
      ([let fresh = let uid = ref 0 in fun () -> …]), one inside a nested module or a [let%ext]
      payload -- unless that binding is [exempt];
    - a tracked binding the sibling [.mli] exports as a [val], since its writers outside the file
      are not read ([Row.mutation_serial] is exported as a function for that reason);
    - an attribute nothing consumes, a malformed one, two on one binding, a stale [exempt] (no write
      of the binding would be a finding without it), a stale [boundary] (no write in the function
      would be), a [fingerprinted] binding [mutated_since] does not read.

    {1 What it deliberately does not read}

    - Branch dominance: a function that notes on one path and writes on another passes. The rule is
      "the definition counts the session's writes", not a control-flow proof.
    - Values whose mutability comes from their type with no allocation call at the head: record
      literals with mutable fields, array literals, objects, a mutable value some function returns.
      Mutable record fields in general ([Shape.t] is private behind guarded setters, gh-ocannl-1506;
      [update_step.unsafe_projections] and delayed references are gh-ocannl-903's field-type
      snapshot).
    - Modules other than the two named: [tensor.ml]'s session records, notably.
    - [Obj] and other escapes from the type system. *)

open Base
open Ppxlib

type kind = Ref | Container of string

type binding_class =
  | Plain
  | Exempt of string
  | Fingerprinted  (** A tracked binding, with its in-place classification. *)

type binding = { name : string; line : int; kind : kind; cls : binding_class }

type report = {
  inventory : string list;  (** What the scan classified, without positions: one line each. *)
  findings : string list;  (** Each refusal, as [file:line: message]. *)
}

let container_modules =
  [
    [ "Hashtbl" ];
    [ "Hashtbl"; "Poly" ];
    [ "Hash_set" ];
    [ "Hash_set"; "Poly" ];
    [ "Queue" ];
    [ "Stack" ];
    [ "Array" ];
  ]

let ref_reads = [ [ "!" ] ]
let ref_writes = [ [ ":=" ]; [ "incr" ]; [ "decr" ]; [ "Int"; "incr" ]; [ "Int"; "decr" ] ]

(** Per container family (the first component of its creation module): the functions that only read
    it, and the functions that write it. *)
let family_functions =
  [
    ( "Hashtbl",
      ( [
          "mem";
          "find";
          "find_exn";
          "find_multi";
          "find_and_call";
          "length";
          "is_empty";
          "data";
          "keys";
          "to_alist";
          "iter";
          "iteri";
          "iter_keys";
          "fold";
          "exists";
          "existsi";
          "for_all";
          "for_alli";
          "count";
          "counti";
        ],
        [
          "set";
          "add";
          "add_exn";
          "add_multi";
          "remove";
          "remove_multi";
          "change";
          "update";
          "update_and_return";
          "find_or_add";
          "findi_or_add";
          "find_and_remove";
          "clear";
          "filter_inplace";
          "filteri_inplace";
          "filter_keys_inplace";
          "map_inplace";
          "mapi_inplace";
          "filter_map_inplace";
          "filter_mapi_inplace";
          "incr";
          "decr";
          "replace";
        ] ) );
    ( "Hash_set",
      ( [
          "mem";
          "length";
          "is_empty";
          "to_list";
          "to_array";
          "iter";
          "fold";
          "exists";
          "for_all";
          "count";
          "find";
        ],
        [
          "add";
          "strict_add";
          "strict_add_exn";
          "remove";
          "strict_remove";
          "strict_remove_exn";
          "clear";
          "filter_inplace";
        ] ) );
    ( "Queue",
      ( [
          "length";
          "is_empty";
          "peek";
          "peek_exn";
          "top";
          "iter";
          "fold";
          "to_list";
          "exists";
          "for_all";
        ],
        [ "enqueue"; "dequeue"; "dequeue_exn"; "add"; "push"; "take"; "pop"; "clear" ] ) );
    ( "Stack",
      ( [ "length"; "is_empty"; "top"; "top_exn"; "iter"; "fold"; "to_list" ],
        [ "push"; "pop"; "pop_exn"; "clear" ] ) );
    ( "Array",
      ( [ "get"; "unsafe_get"; "length"; "iter"; "iteri"; "fold"; "exists"; "for_all"; "to_list" ],
        [ "set"; "unsafe_set"; "fill"; "blit"; "sort"; "swap" ] ) );
  ]

let kind_name = function Ref -> "ref" | Container m -> m

let rec flatten = function
  | Longident.Lident s -> Some [ s ]
  | Ldot (p, s) -> Option.map (flatten p) ~f:(fun p -> p @ [ s ])
  | Lapply _ -> None

(* A [Stdlib.]/[Base.] qualifier says nothing about which function it is. *)
let unqualified = function ("Stdlib" | "Base") :: (_ :: _ as rest) -> rest | path -> path

let rec strip (e : expression) =
  match e.pexp_desc with Pexp_constraint (e, _) | Pexp_coerce (e, _, _) -> strip e | _ -> e

let allocation (e : expression) =
  match (strip e).pexp_desc with
  | Pexp_apply ({ pexp_desc = Pexp_ident { txt; _ }; _ }, _) -> (
      match Option.map (flatten txt) ~f:unqualified with
      | Some [ "ref" ] -> `Alloc Ref
      | Some path -> (
          match List.rev path with
          | ("create" | "make") :: (_ :: _ as rev_module) ->
              let m = List.rev rev_module in
              if List.mem container_modules m ~equal:(List.equal String.equal) then
                `Alloc (Container (List.hd_exn m))
              else `Unknown (String.concat ~sep:"." path)
          | _ -> `No)
      | None -> `No)
  | _ -> `No

let rec is_function (e : expression) =
  match e.pexp_desc with
  | Pexp_function _ -> true
  | Pexp_constraint (e, _) | Pexp_coerce (e, _, _) | Pexp_newtype (_, e) -> is_function e
  | _ -> false

let binding_name (vb : value_binding) =
  match vb.pvb_pat.ppat_desc with
  | Ppat_var { txt; _ } | Ppat_constraint ({ ppat_desc = Ppat_var { txt; _ }; _ }, _) -> Some txt
  | _ -> None

let line_of (loc : Location.t) = loc.loc_start.pos_lnum
let is_ours (a : attribute) = String.is_prefix a.attr_name.txt ~prefix:"mutation_serial"

type attr = A_exempt of string | A_fingerprinted | A_boundary of string

let word_count s =
  List.length
    (String.split_on_chars s ~on:[ ' '; '\t'; '\n' ] |> List.filter ~f:(Fn.non String.is_empty))

(* [Ok attr], or the reason the attribute is malformed. *)
let parse_attr (a : attribute) =
  let reason () =
    match a.attr_payload with
    | PStr
        [
          {
            pstr_desc = Pstr_eval ({ pexp_desc = Pexp_constant (Pconst_string (s, _, _)); _ }, _);
            _;
          };
        ]
      when word_count s > 1 ->
        Ok s
    | _ -> Error "its payload must be one string literal giving a reason of more than one word"
  in
  match a.attr_name.txt with
  | "mutation_serial.exempt" -> Result.map (reason ()) ~f:(fun s -> A_exempt s)
  | "mutation_serial.boundary" -> Result.map (reason ()) ~f:(fun s -> A_boundary s)
  | "mutation_serial.fingerprinted" -> (
      match a.attr_payload with
      | PStr [] -> Ok A_fingerprinted
      | _ -> Error "it takes no payload: the relationship to mutated_since is its reason")
  | other ->
      Error
        (Printf.sprintf
           "%s is not one of mutation_serial.exempt, mutation_serial.fingerprinted, \
            mutation_serial.boundary"
           other)

type denotation = [ `Tracked of binding | `Note | `Local ]

type frame = {
  fname : string;
  root : string;
  boundary : (string * Location.t) option;
  mutable noted : bool;
  mutable writes : (binding * int) list;
}

let scan ~file ?interface source =
  let findings = ref [] in
  let finding line fmt =
    Printf.ksprintf
      (fun msg -> findings := Printf.sprintf "%s:%d: %s" file line msg :: !findings)
      fmt
  in
  let lexbuf = Lexing.from_string source in
  Lexing.set_filename lexbuf file;
  let structure = Parse.implementation lexbuf in
  (* Every attribute of ours, by location; consumption is recorded against it. *)
  let ours = Hashtbl.create (module Int) in
  let collector =
    object
      inherit Ast_traverse.iter as super

      method! attribute a =
        if is_ours a then Hashtbl.set ours ~key:a.attr_loc.loc_start.pos_cnum ~data:(a, ref false);
        super#attribute a
    end
  in
  collector#structure structure;
  let consume (a : attribute) =
    Option.iter (Hashtbl.find ours a.attr_loc.loc_start.pos_cnum) ~f:(fun (_, used) -> used := true)
  in
  (* The one classification of a binding or function, consumed; [None] when it has none. *)
  let classification_of attrs =
    match List.filter attrs ~f:is_ours with
    | [] -> None
    | [ a ] -> (
        match parse_attr a with
        | Ok parsed -> Some (parsed, a)
        | Error why ->
            consume a;
            finding (line_of a.attr_loc) "malformed attribute [@@%s]: %s" a.attr_name.txt why;
            None)
    | a :: _ as several ->
        List.iter several ~f:consume;
        finding (line_of a.attr_loc) "%d mutation_serial attributes on one binding: give it one"
          (List.length several);
        None
  in
  (* {2 Discovery} *)
  let tracked = Hashtbl.create (module Int) in
  let note_binding = ref None in
  let exempt_used = Hash_set.create (module String) in
  let bindings = ref [] in
  (* Exempt state the scan does not resolve: hidden, nested, or bound by a pattern. *)
  let unread_exempt = ref [] in
  let hidden_allocations (vb : value_binding) =
    let found = ref [] in
    let walker =
      object
        inherit Ast_traverse.iter as super

        method! expression e =
          match e.pexp_desc with
          | Pexp_function _ -> ()
          | _ ->
              (if not (phys_equal e (strip vb.pvb_expr)) then
                 match allocation e with
                 | `Alloc _ -> found := line_of e.pexp_loc :: !found
                 | _ -> ());
              super#expression e
      end
    in
    walker#expression vb.pvb_expr;
    !found
  in
  let rec discover ~nested items =
    List.iter items ~f:(fun item ->
        match item.pstr_desc with
        | Pstr_value (_, vbs) -> List.iter vbs ~f:(discover_binding ~nested)
        | Pstr_extension ((_, PStr items), _) -> discover ~nested:true items
        | Pstr_module { pmb_expr; _ } -> discover_module pmb_expr
        | Pstr_recmodule mbs -> List.iter mbs ~f:(fun mb -> discover_module mb.pmb_expr)
        | _ -> ())
  and discover_module me =
    match me.pmod_desc with
    | Pmod_structure items -> discover ~nested:true items
    | Pmod_constraint (me, _) | Pmod_functor (_, me) -> discover_module me
    | _ -> ()
  and discover_binding ~nested vb =
    let line = line_of vb.pvb_loc in
    let name = binding_name vb in
    let head = allocation vb.pvb_expr in
    let hidden = hidden_allocations vb in
    let cls = classification_of vb.pvb_attributes in
    let exempt = match cls with Some (A_exempt _, a) -> Some a | _ -> None in
    let consume_exempt () = Option.iter exempt ~f:consume in
    (match (name, nested) with
    | Some "note_mutation", false -> note_binding := Some vb.pvb_loc.loc_start.pos_cnum
    | _ -> ());
    (match head with
    | `Unknown path ->
        finding line
          "%s is not a constructor the scan reads (the closed list: ref, and create/make of \
           Hashtbl, Hashtbl.Poly, Hash_set, Hash_set.Poly, Queue, Stack, Array); extend the list \
           or bind something it reads"
          path
    | `No -> ()
    | `Alloc kind -> (
        match (name, nested) with
        | Some name, false ->
            let bcls =
              match cls with
              | Some (A_exempt reason, a) ->
                  consume a;
                  Exempt reason
              | Some (A_fingerprinted, a) ->
                  consume a;
                  (match kind with
                  | Ref -> ()
                  | Container _ -> finding line "%s: only a ref can be fingerprinted" name);
                  Fingerprinted
              | Some (A_boundary _, _) | None -> Plain
            in
            let b = { name; line; kind; cls = bcls } in
            Hashtbl.set tracked ~key:vb.pvb_loc.loc_start.pos_cnum ~data:b;
            bindings := b :: !bindings
        | _ when Option.is_some exempt ->
            consume_exempt ();
            unread_exempt := Option.value name ~default:"<pattern>" :: !unread_exempt
        | None, _ -> finding line "a mutable allocation bound by a pattern: bind it to a name"
        | Some name, true ->
            finding line
              "%s: module-level state inside a nested module or extension payload is not read; \
               hoist it to the top level or exempt it"
              name));
    if not (List.is_empty hidden) then
      if Option.is_some exempt then (
        consume_exempt ();
        unread_exempt := Option.value name ~default:"<pattern>" :: !unread_exempt)
      else
        match name with
        | None -> finding line "a mutable allocation bound by a pattern: bind it to a name"
        | Some name ->
            finding line
              "%s: module-level state hidden under a local let of a module-level value is not \
               read; bind it at the top level or exempt the enclosing binding"
              name
  in
  discover ~nested:false structure;
  (* {2 Resolution} *)
  let classified = Hash_set.create (module Int) in
  let fingerprint_reads = Hash_set.create (module String) in
  let frames : frame list ref = ref [] in
  let fun_depth = ref 0 in
  let boundaries = ref [] in
  let judge (frame : frame) =
    let boundary_used = ref false in
    Option.iter frame.boundary ~f:(fun (reason, _) ->
        boundaries := Printf.sprintf "%s: boundary %s -- %s" file frame.fname reason :: !boundaries);
    List.iter (List.rev frame.writes) ~f:(fun (b, line) ->
        match b.cls with
        | Fingerprinted -> ()
        | _ when frame.noted -> ()
        | Exempt _ -> Hash_set.add exempt_used b.name
        | Plain when Option.is_some frame.boundary -> boundary_used := true
        | Plain ->
            finding line "%s writes %s without note_mutation () (nor a classification)" frame.fname
              b.name);
    match frame.boundary with
    | Some (_, loc) when not !boundary_used ->
        finding (line_of loc)
          "stale boundary on %s: none of its writes would need it -- remove the attribute"
          frame.fname
    | _ -> ()
  in
  let scanner =
    object (self)
      inherit [denotation, [ `Local ]] Lexical_scope.scoped as super
      method local = `Local

      (* Every module binding the file makes is local: a [Row] bound anywhere in scope -- a module,
         a functor parameter, an unpack -- is not the library's. *)
      method module_path _ _ = Some `Local
      method! shadowed = Some `Local

      method! binding_denotes _ vb =
        let key = vb.pvb_loc.loc_start.pos_cnum in
        match Hashtbl.find tracked key with
        | Some b -> `Tracked b
        | None -> if Option.equal Int.equal !note_binding (Some key) then `Note else `Local

      method tracked env (e : expression) =
        match e.pexp_desc with
        | Pexp_ident { txt = Lident x; loc } -> (
            match Lexical_scope.lookup env x with Some (`Tracked b) -> Some (b, loc) | _ -> None)
        | _ -> None

      method is_note env (e : expression) =
        match e.pexp_desc with
        | Pexp_ident { txt = Ldot (Lident "Row", "note_mutation"); _ } ->
            not (Map.mem env.Lexical_scope.modules "Row")
        | Pexp_ident { txt = Lident "note_mutation"; _ } -> (
            match Lexical_scope.lookup env "note_mutation" with Some `Note -> true | _ -> false)
        | _ -> false

      method record_write b loc =
        match !frames with
        | frame :: _ when !fun_depth > 0 -> frame.writes <- (b, line_of loc) :: frame.writes
        | _ ->
            finding (line_of loc)
              "writes %s at module initialization, under no function: not a session write the scan \
               can account for"
              b.name

      method! expression env e =
        (match e.pexp_desc with
        | Pexp_apply
            (fn, [ (Nolabel, { pexp_desc = Pexp_construct ({ txt = Lident "()"; _ }, None); _ }) ])
          when self#is_note env fn ->
            List.iter !frames ~f:(fun frame -> frame.noted <- true)
        | Pexp_apply ({ pexp_desc = Pexp_ident { txt = fn; _ }; _ }, args) -> (
            match List.find args ~f:(fun (label, _) -> Poly.equal label Nolabel) with
            | None -> ()
            | Some (_, arg) -> (
                match self#tracked env arg with
                | None -> ()
                | Some (b, loc) -> (
                    let path = Option.value_map (flatten fn) ~f:unqualified ~default:[] in
                    let verdict =
                      match b.kind with
                      | Ref ->
                          if List.mem ref_reads path ~equal:(List.equal String.equal) then `Read
                          else if List.mem ref_writes path ~equal:(List.equal String.equal) then
                            `Write
                          else `Unlisted
                      | Container family -> (
                          match
                            (path, List.Assoc.find family_functions family ~equal:String.equal)
                          with
                          | head :: (_ :: _ as rest), Some (reads, writes)
                            when String.equal head family ->
                              let fname = List.last_exn rest in
                              if List.mem reads fname ~equal:String.equal then `Read
                              else if List.mem writes fname ~equal:String.equal then `Write
                              else `Unlisted
                          | _ -> `Unlisted)
                    in
                    match verdict with
                    | `Read -> (
                        Hash_set.add classified loc.loc_start.pos_cnum;
                        match !frames with
                        | frame :: _ when String.equal frame.root "mutated_since" ->
                            Hash_set.add fingerprint_reads b.name
                        | _ -> ())
                    | `Write ->
                        Hash_set.add classified loc.loc_start.pos_cnum;
                        self#record_write b loc
                    | `Unlisted -> ())))
        | _ -> ());
        match e.pexp_desc with
        | Pexp_function _ ->
            Int.incr fun_depth;
            let result = super#expression env e in
            Int.decr fun_depth;
            result
        | _ -> super#expression env e

      method! ident env { txt; loc } =
        match txt with
        | Lident x -> (
            match Lexical_scope.lookup env x with
            | Some (`Tracked b) when not (Hash_set.mem classified loc.loc_start.pos_cnum) ->
                finding (line_of loc)
                  "%s (%s) is used in a form the scan does not classify: only the listed reads and \
                   writes of its own kind, with the binding as the first positional argument"
                  b.name (kind_name b.kind)
            | _ -> ())
        | _ -> ()

      method! value_binding env vb =
        if is_function vb.pvb_expr || List.is_empty !frames then (
          let fname = Option.value (binding_name vb) ~default:"<pattern>" in
          let boundary =
            if Hashtbl.mem tracked vb.pvb_loc.loc_start.pos_cnum then None
            else
              match List.filter vb.pvb_attributes ~f:is_ours with
              | [ ({ attr_name = { txt = "mutation_serial.boundary"; _ }; _ } as a) ] -> (
                  match parse_attr a with
                  | Ok (A_boundary reason) ->
                      consume a;
                      if is_function vb.pvb_expr then Some (reason, a.attr_loc) else None
                  | _ -> None)
              | _ -> None
          in
          let root = match List.last !frames with Some f -> f.root | None -> fname in
          let frame = { fname; root; boundary; noted = false; writes = [] } in
          frames := frame :: !frames;
          let result = super#value_binding env vb in
          frames := List.tl_exn !frames;
          judge frame;
          result)
        else super#value_binding env vb
    end
  in
  ignore
    (scanner#structure { Lexical_scope.frames = []; modules = Map.empty (module String) } structure
      : structure);
  (* Boundary attributes are parsed again at their function; a malformed one reports once here. *)
  Hashtbl.iter ours ~f:(fun (a, used) ->
      if not !used then
        match parse_attr a with
        | Error why ->
            finding (line_of a.attr_loc) "malformed attribute [@@%s]: %s" a.attr_name.txt why
        | Ok _ ->
            finding (line_of a.attr_loc)
              "[@@%s] is attached to nothing it classifies: exempt and fingerprinted belong on a \
               tracked binding, boundary on a let-bound function"
              a.attr_name.txt);
  let bindings = List.rev !bindings in
  List.iter bindings ~f:(fun b ->
      match b.cls with
      | Fingerprinted when not (Hash_set.mem fingerprint_reads b.name) ->
          finding b.line "%s is fingerprinted, but mutated_since does not read !%s" b.name b.name
      | _ -> ());
  (* An exempt tracked binding whose writes would all be accounted for anyway is stale. *)
  Hashtbl.iter tracked ~f:(fun b ->
      match b.cls with
      | Exempt _ when not (Hash_set.mem exempt_used b.name) ->
          finding b.line "stale exemption on %s: none of its writes would need it" b.name
      | _ -> ());
  Option.iter interface ~f:(fun interface ->
      let lexbuf = Lexing.from_string interface in
      Lexing.set_filename lexbuf (file ^ "i");
      let exported =
        List.filter_map (Parse.interface lexbuf) ~f:(fun item ->
            match item.psig_desc with Psig_value vd -> Some vd.pval_name.txt | _ -> None)
      in
      Hashtbl.iter tracked ~f:(fun b ->
          if List.mem exported b.name ~equal:String.equal then
            finding b.line
              "%s is exported by the interface: its writers outside this file are not read; export \
               a function instead"
              b.name));
  let inventory =
    List.map bindings ~f:(fun b ->
        Printf.sprintf "%s: %s (%s) -- %s" file b.name (kind_name b.kind)
          (match b.cls with
          | Plain -> "accounted: every write noted or at a boundary"
          | Exempt reason -> "exempt: " ^ reason
          | Fingerprinted -> "fingerprinted by mutated_since"))
    @ List.map (List.rev !unread_exempt) ~f:(fun name ->
        Printf.sprintf "%s: %s (unread state) -- exempt" file name)
    @ List.rev !boundaries
  in
  {
    inventory;
    findings =
      List.rev !findings
      |> List.dedup_and_sort ~compare:(fun a b ->
          let line s =
            Option.value_map (List.nth (String.split s ~on:':') 1) ~f:Int.of_string ~default:0
          in
          match Int.compare (line a) (line b) with 0 -> String.compare a b | c -> c);
  }
