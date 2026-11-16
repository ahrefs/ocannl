(* Reading a kernel's scope definitions back out of its text, for the tests that hold
   [Low_level.Access_fold]'s [Emitted] walk to what renders (gh-ocannl-2213, gh-ocannl-2250).

   A local spells its scope id first ([v<id>_], {!Ir.Low_level.scope_local_name}) and a declaration
   starts with its type, so a line assigning a local is one starting with the local and an [=]: a
   wrapped operand line can start with a read of it. A line naming the local that neither assigns
   nor declares it reads it. What this reads is the source text alone; what it does not: which block
   a line sits in, or anything the text names other than a scope local. *)

open Base
module LL = Ir.Low_level

let local_prefix id = "v" ^ Int.to_string id ^ "_"
let ident_char ch = Char.is_alphanum ch || Char.equal ch '_'

(* [l] (left-stripped) assigns the local spelled [prefix]... *)
let assigns ~prefix l =
  match String.chop_prefix l ~prefix with
  | None -> false
  | Some rest ->
      let rest = String.lstrip ~drop:ident_char rest in
      String.is_prefix rest ~prefix:" = "

(* The positions where [prefix] starts a whole identifier of [l]. *)
let mentions ~prefix l =
  String.substr_index_all l ~may_overlap:false ~pattern:prefix
  |> List.filter ~f:(fun at -> at = 0 || not (ident_char l.[at - 1]))

(* [l] (left-stripped) declares the local spelled [prefix]...: type words, the local, then an
   initializer or the end of the declaration. *)
let declares ~prefix l =
  match mentions ~prefix l with
  | [ at ] when at > 0 ->
      let before = String.prefix l at and after = String.drop_prefix l at in
      let after = String.lstrip ~drop:ident_char after in
      String.for_all before ~f:(fun ch -> ident_char ch || Char.equal ch ' ')
      && (String.is_prefix after ~prefix:" =" || String.is_prefix after ~prefix:";")
  | _ -> false

let stripped_lines source =
  List.mapi (String.split_lines source) ~f:(fun k l -> (k, String.lstrip l))

(* The line of [source] holding the last assignment of scope [id]'s local, if any. *)
let last_assignment source id =
  let prefix = local_prefix id in
  List.rev (stripped_lines source)
  |> List.find ~f:(fun (_, l) -> assigns ~prefix l)
  |> Option.map ~f:fst

(** The scope bodies of [llc] as [Access_fold] walks them under [local_scopes], each when its walk
    ends -- a body ends after the definitions nested in it -- paired with the line of [source]
    holding its last assignment ([None]: no line assigns it). Pass the tree that renders: under
    [Emitted], the one {!Ir.C_syntax.scope_order_oracles} recorded (after the exact-scope collapse),
    so that every body it holds is in the text. *)
let body_ends local_scopes (llc : LL.t) source =
  let policy : LL.Access_fold.policy =
    {
      discarded_operands = Skip;
      gated_operands = Visit;
      dead_loops = Visit;
      local_scopes;
      guards = Ignore;
      scan_implicit = Visit;
    }
  in
  let after_scalar _ acc = function
    | LL.Local_scope { id; _ } -> id.LL.scope_id :: acc
    | _ -> acc
  in
  let visits =
    List.rev
      (LL.Access_fold.fold ~policy
         ~hooks:{ (LL.Access_fold.hooks ()) with after_scalar }
         ~init:[] llc)
  in
  List.map visits ~f:(fun id -> (id, last_assignment source id))

(** Every body found in the text, each once, and in text order. *)
let in_text_order ends =
  match Option.all (List.map ends ~f:snd) with
  | None -> false
  | Some ks ->
      List.is_sorted_strictly ks ~compare:Int.compare
      && not (List.contains_dup (List.map ends ~f:fst) ~compare:Int.compare)

(** The lines of [source] reading scope [id]'s local ahead of its last assignment: empty where the
    text defines the local before anything reads it. A local the text never assigns has every read
    ahead of it. *)
let reads_ahead_of_definition source id =
  let prefix = local_prefix id in
  let lines = stripped_lines source in
  let last = Option.value (last_assignment source id) ~default:Int.max_value in
  List.filter_map lines ~f:(fun (k, l) ->
      if
        k < last
        && (not (List.is_empty (mentions ~prefix l)))
        && (not (assigns ~prefix l))
        && not (declares ~prefix l)
      then Some k
      else None)

(** Whether [source] names scope [id]'s local at all. *)
let names_local source id =
  let prefix = local_prefix id in
  List.exists (String.split_lines source) ~f:(fun l -> not (List.is_empty (mentions ~prefix l)))
