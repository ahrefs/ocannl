(* gh-ocannl-1506: Shape.t is a private record, so the closed-shape guard's setters are the only way
   to write a shape. gh-ocannl-1573 extended the same to the rest of gh-ocannl-903's frontier: an
   update step (its projections), a capture reference ([var]) and the [Ir.Indexing.variable_ref] it
   carries ([solved_dim], [solved_sym]). gh-ocannl-1624 closed the frontier: a static symbol's
   fields (its roles [used_as_extent] / [used_as_slice] above all), a tensor node's delayed
   precision, and the Indexing setters, which take the shape writer only [Shape] holds. That is a
   property of the INTERFACE, which nothing executed can observe: this control hands real ocamlc
   programs against the built Shape interface. Reads and pattern matches must compile; an assignment
   to each mutable field, a record literal, a record update and the retired [t_of_sexp] must each be
   rejected with their own diagnostic. Every rejected leg that writes, builds or calls an existing
   name compiled on the base, where the record was public; the two that name what the implementation
   keeps hidden (the writer's constructor, the precision cell's raw setter) were unbound there too,
   and pin that they stay so. A failed compiler launch or a missing include directory cannot pass as
   a rejection: the positive leg runs through the same path, and each negative leg names the
   diagnostic it expects. *)
open Base
open Stdio
open Verdict.Claims

let compile ~compiler ~include_dirs code =
  let source = Stdlib.Filename.temp_file "shape_private_control" ".ml" in
  let output = Stdlib.Filename.chop_suffix source ".ml" ^ ".cmo" in
  let log = source ^ ".log" in
  Exn.protect
    ~f:(fun () ->
      Out_channel.write_all source ~data:code;
      let fd = Unix.openfile log [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_TRUNC ] 0o600 in
      let pid =
        Exn.protect
          ~f:(fun () ->
            Unix.create_process compiler
              (Array.of_list
                 ([ compiler; "-color"; "never"; "-w"; "-a" ]
                 @ List.concat_map include_dirs ~f:(fun dir -> [ "-I"; dir ])
                 @ [ "-c"; "-o"; output; source ]))
              Unix.stdin fd fd)
          ~finally:(fun () -> Unix.close fd)
      in
      let _, status = Unix.waitpid [] pid in
      (status, In_channel.read_all log))
    ~finally:(fun () ->
      List.iter
        [ source; output; Stdlib.Filename.chop_suffix source ".ml" ^ ".cmi"; log ]
        ~f:(fun path -> if Stdlib.Sys.file_exists path then Stdlib.Sys.remove path))

(* The words of the log's text, punctuation and quoting dropped: how a release delimits a name is
   presentation (gh-ocannl-1223). *)
let words text =
  String.split_on_chars text ~on:[ ' '; '\t'; '\n'; '\r'; ':'; '"'; '.'; ','; '('; ')' ]
  |> List.filter ~f:(Fn.non String.is_empty)

let contains_run text run =
  let rec go = function
    | [] -> false
    | _ :: rest as ws -> List.is_prefix ws ~prefix:run ~equal:String.equal || go rest
  in
  go (words text)

let prelude =
  "type shape = Ocannl_tensor.Shape.t\n\
   type step = Ocannl_tensor.Shape.update_step\n\
   type dref = Ocannl_tensor.Shape.delayed_var_ref\n\
   type vref = Ir.Indexing.variable_ref\n\
   type ssym = Ir.Indexing.static_symbol\n\
   type tnode = Ir.Tnode.t\n"

let private_write field = [ "Cannot"; "assign"; "field"; field; "of"; "the"; "private"; "type" ]
let private_construction = [ "Cannot"; "create"; "values"; "of"; "the"; "private"; "type" ]

let () =
  let compiler = Stdlib.Sys.argv.(1) in
  (* The directories of the interfaces the snippets reach: Shape's own, Ir's (for the capture
     reference's [variable_ref] and a step's projections), and Base's for field types. *)
  let include_dirs =
    List.map (List.drop (Array.to_list Stdlib.Sys.argv) 2) ~f:Stdlib.Filename.dirname
  in
  let positive, positive_log =
    compile ~compiler ~include_dirs
      (prelude
     ^ "let read (sh : shape) = sh.closed && sh.id >= 0 && Option.is_none sh.discarded_by\n\
        let matched (sh : shape) = match sh with { closed; batch = _; _ } -> closed\n")
  in
  if not (Poly.equal positive (Unix.WEXITED 0)) then eprintf "%s" positive_log;
  p "reading and pattern-matching a shape's fields compiles" (Poly.equal positive (Unix.WEXITED 0));
  let frontier, frontier_log =
    compile ~compiler ~include_dirs
      (prelude
     ^ "let read_step (s : step) = Option.is_none s.unsafe_projections && s.neutral_elem = None\n\
        let match_step (s : step) = match s with { unsafe_projections = None; _ } -> true | _ -> \
        false\n\
        let read_ref (r : dref) =\n\
       \  match r with\n\
       \  | { var = `Not_set_yet; var_ref = { solved_dim = None; _ }; _ } -> true\n\
       \  | _ -> Option.is_some r.var_ref.solved_sym\n\
        let read_vref (v : vref) = v.ref_label\n\
        let fresh_ref : dref = Ocannl_tensor.Shape.get_variable_ref \"d\"\n\
        let fresh_vref : vref = Ir.Indexing.variable_ref \"d\"\n\
        let make = Ocannl_tensor.Shape.make_update_step\n\
        let read_ssym (s : ssym) = s.used_as_extent || s.used_as_slice || Option.is_some \
        s.static_range\n\
        let match_ssym (s : ssym) = match s with { used_as_extent = true; _ } -> true | _ -> false\n\
        let fresh_ssym : ssym =\n\
       \  Ir.Indexing.make_static_symbol ~static_range:2 ~used_as:`Extent (Ir.Indexing.get_symbol \
        ())\n\
        let range (s : ssym) = Ir.Indexing.set_static_range s (Some 3)\n\
        let read_prec (tn : tnode) = Ir.Tnode.Prec_cell.get tn.delayed_prec_unsafe\n")
  in
  if not (Poly.equal frontier (Unix.WEXITED 0)) then eprintf "%s" frontier_log;
  p
    "reading and matching an update step, a capture reference and its variable_ref, a static \
     symbol and a node's delayed precision compiles, and so do their public constructors"
    (Poly.equal frontier (Unix.WEXITED 0));
  (* By default the diagnostic's words must follow [Error] directly. [~anywhere:true] matches them
     anywhere in the log, apart from an [Error] that is still required: for a refusal printed among
     other lines whose wording before the run varies across compiler releases. *)
  let rejected ?(anywhere = false) name code expected =
    let status, log = compile ~compiler ~include_dirs (prelude ^ code) in
    let matched =
      if anywhere then contains_run log [ "Error" ] && contains_run log expected
      else contains_run log ("Error" :: expected)
    in
    let ok = Poly.equal status (Unix.WEXITED 2) && matched in
    if not ok then eprintf "%s:\n%s\n" name log;
    p name ok
  in
  List.iter
    [
      "batch";
      "input";
      "output";
      "batch_padding";
      "input_padding";
      "output_padding";
      "padding_elem";
      "closed";
      "discarded_by";
    ] ~f:(fun field ->
      rejected
        (Printf.sprintf "assigning the %s field is rejected as a private-type write" field)
        (Printf.sprintf "let write (sh : shape) = sh.%s <- sh.%s\n" field field)
        [ "Cannot"; "assign"; "field"; field; "of"; "the"; "private"; "type" ]);
  rejected "a record update of a shape is rejected as a private-type construction"
    "let copy (sh : shape) : shape = { sh with id = 0 }\n"
    [ "Cannot"; "create"; "values"; "of"; "the"; "private"; "type" ];
  rejected "a record literal of a shape is rejected as a private-type construction"
    "let fresh (sh : shape) : shape =\n\
    \  { batch = sh.batch; input = sh.input; output = sh.output; batch_padding = None;\n\
    \    input_padding = None; output_padding = None; padding_elem = None; closed = false;\n\
    \    discarded_by = None; id = 0; debug_name = \"forged\" }\n"
    [ "Cannot"; "create"; "values"; "of"; "the"; "private"; "type" ];
  (* gh-ocannl-1573: the rest of the frontier. *)
  rejected "assigning an update step's projections is rejected as a private-type write"
    "let write (s : step) = s.unsafe_projections <- None\n"
    (private_write "unsafe_projections");
  rejected "a record update of an update step is rejected as a private-type construction"
    "let copy (s : step) : step = { s with unsafe_projections = None }\n" private_construction;
  rejected "a record literal of an update step is rejected as a private-type construction"
    "let forge (s : step) : step =\n\
    \  { shape = s.shape; logic = s.logic; id = s.id; unsafe_projections = None; neutral_elem = \
     None }\n"
    private_construction;
  rejected "assigning a capture reference's binding is rejected as a private-type write"
    "let write (r : dref) = r.var <- `Not_set_yet\n" (private_write "var");
  rejected "a record update of a capture reference is rejected as a private-type construction"
    "let copy (r : dref) : dref = { r with var = `Not_set_yet }\n" private_construction;
  rejected "a record literal of a capture reference is rejected as a private-type construction"
    "let forge (r : dref) : dref = { var_ref = r.var_ref; var = `Not_set_yet; uid = 0 }\n"
    private_construction;
  rejected "assigning a capture reference's solved dimension is rejected as a private-type write"
    "let write (r : dref) = r.var_ref.solved_dim <- Some 3\n" (private_write "solved_dim");
  rejected "assigning a variable_ref's solved symbol is rejected as a private-type write"
    "let write (v : vref) = v.solved_sym <- None\n" (private_write "solved_sym");
  rejected "a record literal of a variable_ref is rejected as a private-type construction"
    "let forge : vref = { ref_label = \"x\"; solved_dim = None; solved_sym = None }\n"
    private_construction;
  rejected "a record update of a variable_ref is rejected as a private-type construction"
    "let copy (v : vref) : vref = { v with solved_dim = Some 1 }\n" private_construction;
  (* gh-ocannl-1624: a static symbol's fields, the shape writer, and a node's delayed precision. *)
  List.iter [ "used_as_extent"; "used_as_slice"; "static_range" ] ~f:(fun field ->
      rejected
        (Printf.sprintf "assigning a static symbol's %s is rejected as a private-type write" field)
        (Printf.sprintf "let write (s : ssym) = s.%s <- s.%s\n" field field)
        (private_write field));
  rejected "a record literal of a static symbol is rejected as a private-type construction"
    "let forge (s : ssym) : ssym =\n\
    \  { static_symbol = s.static_symbol; static_range = None; used_as_extent = true; \
     used_as_slice = false }\n"
    private_construction;
  rejected "a record update of a static symbol is rejected as a private-type construction"
    "let copy (s : ssym) : ssym = { s with used_as_extent = true }\n" private_construction;
  (* The setters' first argument is the writer: a call without it is a type error naming it. The
     words before the expected type vary across compiler releases, so the run is matched alone. *)
  let wanting_writer =
    [ "but"; "an"; "expression"; "was"; "expected"; "of"; "type"; "Ir"; "Indexing"; "shape_writer" ]
  in
  rejected ~anywhere:true "solving a variable_ref without the shape writer is a type error"
    "let write (v : vref) = Ir.Indexing.set_solved_dim v 3\n" wanting_writer;
  rejected ~anywhere:true
    "solving a variable_ref to a symbol without the shape writer is a type error"
    "let write (v : vref) (s : ssym) = Ir.Indexing.set_solved_sym v s\n" wanting_writer;
  rejected ~anywhere:true "committing a role without the shape writer is a type error"
    "let write (s : ssym) = Ir.Indexing.commit_role s `Extent\n" wanting_writer;
  rejected "the shape writer has no constructor to forge it with"
    "let forge : Ir.Indexing.shape_writer = Ir.Indexing.Shape_writer\n"
    [ "Unbound"; "constructor"; "Ir"; "Indexing"; "Shape_writer" ];
  rejected "assigning a node's delayed precision is rejected: the field is not mutable"
    "let write (tn : tnode) = tn.delayed_prec_unsafe <- tn.delayed_prec_unsafe\n"
    [ "The"; "record"; "field"; "delayed_prec_unsafe"; "is"; "not"; "mutable" ];
  rejected "a node's precision cell offers no raw setter"
    "let write (tn : tnode) = Ir.Tnode.Prec_cell.set tn.delayed_prec_unsafe (Ir.Tnode.Specified \
     Ir.Ops.half)\n"
    [ "Unbound"; "value"; "Ir"; "Tnode"; "Prec_cell"; "set" ];
  rejected "deserializing a shape is no longer offered"
    "let parse = Ocannl_tensor.Shape.t_of_sexp\n"
    [ "Unbound"; "value"; "Ocannl_tensor"; "Shape"; "t_of_sexp" ]
