(* gh-ocannl-1162: the affine queries against enumerated physical cell addresses.

   [Indexing.Sub_axis] has ONE meaning in the IR: the axis contributes zero to the row-major flat
   offset while keeping its stride, so a [Sub_axis] run followed by a component makes that component
   a flattened index over the run's whole extent (lowering's flat stores, [tensor/row.ml]). The
   query callers also need to say "this component is unknown" — a dynamic access's data-dependent
   axis, a vectorized store's run — and that must never be confused with the flattened meaning.

   Every case here builds small physical buffers and enumerates, for each valuation of the loop
   symbols, the cells an access touches in the renderer's own address arithmetic (Horner over the
   dims, [Sub_axis] adding zero; a dynamic axis taking every value of its dim; a vector run covering
   [length] consecutive flat cells from its base). A query's proven answer — [Disjoint],
   [Same_thread], "separated", "within the box", "covers the box", an exact fiber — must agree with
   that enumeration; a conservative answer is printed, so precision is visible in the golden, but
   never fails. The accesses reach the engine in the form the production callers give them, through
   [Q] below. *)

open Base
module Idx = Ir.Indexing
module Aff = Ir.Affine
open Verdict.Claims

let sym = Idx.get_symbol
let aff terms offset = Idx.affine ~symbols:terms ~offset
let it s = Idx.Iterator s
let fx c = Idx.Fixed_idx c
let sub = Idx.Sub_axis

(* An access as the caller holds it: the IR map plus what the caller knows is NOT a static cell. *)
type acc = {
  map : Idx.axis_index array;
  dyn : int option;  (** The data-dependent axis ([Set_dynamic]/[Get_dynamic]'s [dyn_axis]). *)
  vec : int option;  (** A vectorized store's run length ([Set_from_vec]'s [length]). *)
}

let plain map = { map; dyn = None; vec = None }
let dynamic ~axis map = { map; dyn = Some axis; vec = None }
let vector ~length map = { map; dyn = None; vec = Some length }
let find_range ranges s = List.Assoc.find ranges s ~equal:Idx.equal_symbol
let dedup = List.dedup_and_sort ~compare:Idx.compare_symbol

let mentioned (map : Idx.axis_index array) =
  Array.to_list map
  |> List.concat_map ~f:(function
    | Idx.Iterator s -> [ s ]
    | Idx.Affine { symbols; _ } -> List.map symbols ~f:snd
    | Idx.Concat syms -> syms
    | Idx.Fixed_idx _ | Idx.Sub_axis -> [])
  |> dedup

(* Every valuation of [syms] within [ranges] (inclusive bounds). *)
let envs ranges syms =
  List.fold syms ~init:[ [] ] ~f:(fun acc s ->
      let lo, hi = Option.value_exn (find_range ranges s) in
      List.concat_map acc ~f:(fun env -> List.init (hi - lo + 1) ~f:(fun k -> (s, lo + k) :: env)))

(* The flat cells one instance touches: the renderer's row-major Horner sum
   ([C_syntax.pp_offset_by], [Indexing.reflect_projection]) with [Sub_axis] adding zero. *)
let cells ~dims env (a : acc) : int list =
  let value s = List.Assoc.find_exn env s ~equal:Idx.equal_symbol in
  let values ax =
    if Option.equal Int.equal a.dyn (Some ax) then List.init dims.(ax) ~f:Fn.id
    else
      match a.map.(ax) with
      | Idx.Sub_axis -> [ 0 ]
      | Idx.Fixed_idx c -> [ c ]
      | Idx.Iterator s -> [ value s ]
      | Idx.Affine { symbols; offset } ->
          [ List.fold symbols ~init:offset ~f:(fun acc (c, s) -> acc + (c * value s)) ]
      | Idx.Concat _ -> failwith "no Concat after lowering"
  in
  let bases =
    Array.foldi dims ~init:[ 0 ] ~f:(fun ax bases d ->
        List.concat_map bases ~f:(fun b -> List.map (values ax) ~f:(fun v -> (b * d) + v)))
  in
  match a.vec with
  | None -> bases
  | Some len -> List.concat_map bases ~f:(fun b -> List.init len ~f:(fun k -> b + k))

let intersects xs ys = List.exists xs ~f:(fun x -> List.mem ys x ~equal:Int.equal)
let lookup env s = List.Assoc.find env s ~equal:Idx.equal_symbol

(* {2 How the production callers hand an access to the engine}

   Through the coordinate view ([Affine.view]): the map as the IR holds it, and what the caller does
   not know stated beside it — the dynamic axis, and a vector store's run, as [Run] for the pair
   queries ([Schedule.query_view], the CPU pool's [`Vec] interpretation) and as [Blocks] for the
   separation check ([Low_level.unseparated_thread_write]). The box queries view the raw map. *)
module Q = struct
  let view ~range ~vec ~dims (a : acc) =
    Aff.view ~range ?dyn_axis:a.dyn ?vec:(Option.map a.vec ~f:vec) ~dims a.map

  let pair_conflict ~dims ~range ~dup_left ~dup_right ~pairs left right =
    let view = view ~range ~vec:(fun l -> Aff.Run l) ~dims in
    Aff.pair_conflict ~range ~dup_left ~dup_right ~pairs ~left:(view left) ~right:(view right)

  let separates ~dims ~range ~concurrent ~syms a =
    Aff.separates ~range ~concurrent ~syms
      ~coords:(view ~range ~vec:(fun l -> Aff.Blocks l) ~dims a)

  let within_box ~dims ~range map = Aff.within_box ~range (Aff.view ~dims map)
  let covers_box ~dims ~range map = Aff.covers_box ~range (Aff.view ~dims map)
end

let show_verdict = function
  | Aff.Disjoint -> "Disjoint"
  | Aff.Same_thread -> "Same_thread"
  | Aff.Cross_thread _ -> "Cross_thread"

(* {2 pair_conflict} *)

let oracle_conflict ~dims ~ranges ~dup_left ~dup_right ~pairs (left : acc) (right : acc) =
  let lsyms = mentioned left.map and rsyms = mentioned right.map in
  let shared = List.filter (dedup (lsyms @ rsyms)) ~f:(fun s -> not (dup_left s || dup_right s)) in
  let l_own = List.filter lsyms ~f:dup_left and r_own = List.filter rsyms ~f:dup_right in
  let conflicts =
    List.concat_map (envs ranges shared) ~f:(fun senv ->
        List.concat_map (envs ranges l_own) ~f:(fun lenv ->
            let lcells = cells ~dims (senv @ lenv) left in
            List.filter_map (envs ranges r_own) ~f:(fun renv ->
                if intersects lcells (cells ~dims (senv @ renv) right) then
                  Some (senv @ lenv, senv @ renv)
                else None)))
  in
  if List.is_empty conflicts then `Disjoint
  else if List.is_empty pairs then `Overlap
  else if
    List.for_all conflicts ~f:(fun (lenv, renv) ->
        List.for_all pairs ~f:(fun (p, p') ->
            match (lookup lenv p, lookup renv p') with
            (* A parallel symbol an access does not mention takes every value on that side. *)
            | Some a, Some b -> a = b
            | _ -> false))
  then `Same_thread
  else `Cross_thread

let show_oracle = function
  | `Disjoint -> "disjoint"
  | `Same_thread -> "same-thread"
  | `Cross_thread -> "cross-thread"
  | `Overlap -> "overlap"

let sound_conflict verdict oracle =
  match (verdict, oracle) with
  | Aff.Disjoint, `Disjoint -> true
  | Aff.Disjoint, _ -> false
  | Aff.Same_thread, (`Disjoint | `Same_thread) -> true
  | Aff.Same_thread, (`Cross_thread | `Overlap) -> false
  | Aff.Cross_thread _, _ -> true

(* [ranges] are the loop bounds; [own_left]/[own_right] the symbols each side iterates independently
   (everything else is shared, equal on both sides); [pairs] the thread identity. *)
let conflict ~name ~dims ~ranges ?(own_left = []) ?(own_right = []) ?(pairs = []) left right =
  let range s = find_range ranges s in
  let dup_left s = List.mem own_left s ~equal:Idx.equal_symbol in
  let dup_right s = List.mem own_right s ~equal:Idx.equal_symbol in
  let verdict = Q.pair_conflict ~dims ~range ~dup_left ~dup_right ~pairs left right in
  let oracle = oracle_conflict ~dims ~ranges ~dup_left ~dup_right ~pairs left right in
  let ok = sound_conflict verdict oracle in
  Stdio.printf "%-60s query %-12s oracle %s\n" name (show_verdict verdict) (show_oracle oracle);
  claimf "%s: the pair verdict agrees with the enumerated cells" name ok

let () =
  Stdio.printf "=== pair_conflict ===\n";
  let h = sym () and e = sym () and f = sym () and c = sym () in
  let i = sym () and j = sym () and u = sym () in
  let r lo hi = (lo, hi) in
  (* The oracle is not vacuous: the issue's example pair shares a cell, [8 * 0 + 40 = 32 * 1 +
     8]. *)
  p "control: [Sub_axis; 40] and [1; 8] address one cell of an 8x32 buffer"
    (intersects
       (cells ~dims:[| 8; 32 |] [] (plain [| sub; fx 40 |]))
       (cells ~dims:[| 8; 32 |] [] (plain [| fx 1; fx 8 |])));
  (* Both readings the issue's table rejects are refuted by the enumeration, so a soundness claim
     below can fail: reading the flattened [Sub_axis; 40] as a placeholder (per-axis "no
     information", the pre-view rule) answers [Disjoint] against [[h; e]], and reading a dynamic
     axis followed by [1] as a flattened run (#728's collapse) answers [Disjoint] against [[2;
     1]]. *)
  let refuted ~dims ~ranges ~own_right ~(reading : acc) ~(truth : acc) (other : acc) =
    let range s = find_range ranges s in
    let dup s = List.mem own_right s ~equal:Idx.equal_symbol in
    let verdict =
      Q.pair_conflict ~dims ~range ~dup_left:(fun _ -> false) ~dup_right:dup ~pairs:[] reading other
    in
    not
      (sound_conflict verdict
         (oracle_conflict ~dims ~ranges
            ~dup_left:(fun _ -> false)
            ~dup_right:dup ~pairs:[] truth other))
  in
  p "control: the placeholder reading of a flattened [Sub_axis; 40] is refuted"
    (refuted ~dims:[| 8; 32 |]
       ~ranges:[ (h, r 0 7); (e, r 0 31) ]
       ~own_right:[ h; e ]
       ~reading:(dynamic ~axis:0 [| fx 0; fx 40 |])
       ~truth:(plain [| sub; fx 40 |])
       (plain [| it h; it e |]));
  p "control: the flattened reading of a dynamic axis then [1] is refuted"
    (refuted ~dims:[| 3; 4 |] ~ranges:[] ~own_right:[]
       ~reading:(plain [| sub; fx 1 |])
       ~truth:(dynamic ~axis:0 [| fx 0; fx 1 |])
       (plain [| fx 2; fx 1 |]));
  (* A flattened store against an ordinary access of the same node. The issue's two examples first:
     both pairs share a cell, and the per-axis reading answered [Disjoint] for both. *)
  conflict ~name:"issue: flat [Sub;40] vs [h;e] (8x32)" ~dims:[| 8; 32 |]
    ~ranges:[ (h, r 0 7); (e, r 0 31) ]
    ~own_right:[ h; e ]
    (plain [| sub; fx 40 |])
    (plain [| it h; it e |]);
  conflict ~name:"issue: flat [Sub;2] vs [1;0] (2x2)" ~dims:[| 2; 2 |] ~ranges:[]
    (plain [| sub; fx 2 |])
    (plain [| fx 1; fx 0 |]);
  conflict ~name:"flat [Sub;f] vs row-major [h;e] (8x32)" ~dims:[| 8; 32 |]
    ~ranges:[ (f, r 0 255); (h, r 0 7); (e, r 0 31) ]
    ~own_left:[ f ] ~own_right:[ h; e ]
    (plain [| sub; it f |])
    (plain [| it h; it e |]);
  conflict ~name:"flat [Sub;f<32] vs row 1 [1;e]: disjoint halves" ~dims:[| 2; 32 |]
    ~ranges:[ (f, r 0 31); (e, r 0 31) ]
    ~own_left:[ f ] ~own_right:[ e ]
    (plain [| sub; it f |])
    (plain [| fx 1; it e |]);
  (* The per-axis reading forced [u = e] from the minor axis alone and answered [Same_thread]; the
     writer's thread [u = 4] writes the cell the reader's thread [e = 0] reads in row 1. *)
  conflict ~name:"flat [Sub;u] vs [h;e], thread u~e" ~dims:[| 2; 4 |]
    ~ranges:[ (u, r 0 7); (h, r 0 1); (e, r 0 3) ]
    ~own_left:[ u ] ~own_right:[ h; e ]
    ~pairs:[ (u, e) ]
    (plain [| sub; it u |])
    (plain [| it h; it e |]);
  (* A masked dynamic axis followed by an iterator: unknown, not a flattened index. *)
  conflict ~name:"dynamic axis then iterator vs [i;j], thread j" ~dims:[| 3; 4 |]
    ~ranges:[ (i, r 0 2); (j, r 0 3) ]
    ~own_left:[ j ] ~own_right:[ i; j ]
    ~pairs:[ (j, j) ]
    (dynamic ~axis:0 [| fx 0; it j |])
    (plain [| it i; it j |]);
  conflict ~name:"dynamic axis then [1] vs [2;1]: may meet" ~dims:[| 3; 4 |] ~ranges:[]
    (dynamic ~axis:0 [| fx 0; fx 1 |])
    (plain [| fx 2; fx 1 |]);
  conflict ~name:"dynamic axis then [1] vs [2;2]: never meet" ~dims:[| 3; 4 |] ~ranges:[]
    (dynamic ~axis:0 [| fx 0; fx 1 |])
    (plain [| fx 2; fx 2 |]);
  (* A flattened run whose last component is vector-masked. *)
  conflict ~name:"flat vec [Sub;4c] run 4 vs [h;e] (2x8)" ~dims:[| 2; 8 |]
    ~ranges:[ (c, r 0 3); (h, r 0 1); (e, r 0 7) ]
    ~own_left:[ c ] ~own_right:[ h; e ]
    (vector ~length:4 [| sub; aff [ (4, c) ] 0 |])
    (plain [| it h; it e |]);
  conflict ~name:"flat vec [Sub;4c] run 4 vs cell [1;7]" ~dims:[| 2; 8 |]
    ~ranges:[ (c, r 0 3) ]
    ~own_left:[ c ]
    (vector ~length:4 [| sub; aff [ (4, c) ] 0 |])
    (plain [| fx 1; fx 7 |]);
  (* The trailing vector-lane store: [Row]'s flat projection leaves a trailing unit axis as
     [Sub_axis], so "mask the last component" masked the wrong one. The tail store of a [2x9x1] node
     (base 16, two lanes) writes cells 16-17, i.e. [1;7;0] and [1;8;0]; the per-axis reading saw
     [16] against a 9-wide axis and answered [Disjoint]. *)
  conflict ~name:"trailing-lane tail vec [Sub;16;Sub] run 2 vs [h;e;0]" ~dims:[| 2; 9; 1 |]
    ~ranges:[ (h, r 0 1); (e, r 0 8) ]
    ~own_right:[ h; e ]
    (vector ~length:2 [| sub; fx 16; sub |])
    (plain [| it h; it e; fx 0 |]);
  (* The shape lowering actually emits: the trailing unit axis pinned to [0] rather than [Sub_axis]
     (affine_view_executed runs it) — masked the same wrong way. *)
  conflict ~name:"lowered tail vec [Sub;16;0] run 2 vs [h;e;0]" ~dims:[| 2; 9; 1 |]
    ~ranges:[ (h, r 0 1); (e, r 0 8) ]
    ~own_right:[ h; e ]
    (vector ~length:2 [| sub; fx 16; fx 0 |])
    (plain [| it h; it e; fx 0 |]);
  conflict ~name:"trailing-lane vec [Sub;4c;Sub] run 4 vs [1;e;0], thread c~e" ~dims:[| 2; 8; 1 |]
    ~ranges:[ (c, r 0 3); (e, r 0 7) ]
    ~own_left:[ c ] ~own_right:[ e ]
    ~pairs:[ (c, e) ]
    (vector ~length:4 [| sub; aff [ (4, c) ] 0; sub |])
    (plain [| fx 1; it e; fx 0 |]);
  (* Repeated symbols: the diagonal against a flat cell. *)
  conflict ~name:"diagonal [i;i] vs flat [Sub;5]: off-diagonal" ~dims:[| 3; 3 |]
    ~ranges:[ (i, r 0 2) ]
    ~own_left:[ i ]
    (plain [| it i; it i |])
    (plain [| sub; fx 5 |]);
  conflict ~name:"diagonal [i;i] vs flat [Sub;4]: on-diagonal" ~dims:[| 3; 3 |]
    ~ranges:[ (i, r 0 2) ]
    ~own_left:[ i ]
    (plain [| it i; it i |])
    (plain [| sub; fx 4 |]);
  conflict ~name:"diagonal [i;i] vs flat [Sub;f]" ~dims:[| 3; 3 |]
    ~ranges:[ (i, r 0 2); (f, r 0 8) ]
    ~own_left:[ i ] ~own_right:[ f ]
    (plain [| it i; it i |])
    (plain [| sub; it f |]);
  (* Placeholders on both sides of a same-nest pair. *)
  conflict ~name:"vec [i;4j] run 4 vs itself, thread i" ~dims:[| 3; 8 |]
    ~ranges:[ (i, r 0 2); (j, r 0 1) ]
    ~own_left:[ i; j ] ~own_right:[ i; j ]
    ~pairs:[ (i, i) ]
    (vector ~length:4 [| it i; aff [ (4, j) ] 0 |])
    (vector ~length:4 [| it i; aff [ (4, j) ] 0 |])

(* {2 separates} *)

(* Two instances iterate [concurrent] independently (everything else shared); separation holds when
   every pair of instances sharing a cell agrees on [syms]. *)
let oracle_separates ~dims ~ranges ~concurrent ~syms (a : acc) =
  let syms_a = mentioned a.map in
  let shared = List.filter syms_a ~f:(fun s -> not (concurrent s)) in
  let own = List.filter syms_a ~f:concurrent in
  let clash =
    List.find_map (envs ranges shared) ~f:(fun senv ->
        let insts = List.map (envs ranges own) ~f:(fun env -> (env, cells ~dims (senv @ env) a)) in
        List.find_map insts ~f:(fun (e1, c1) ->
            List.find_map insts ~f:(fun (e2, c2) ->
                let disagree =
                  List.find syms ~f:(fun s ->
                      match (lookup e1 s, lookup e2 s) with Some x, Some y -> x <> y | _ -> true)
                in
                if intersects c1 c2 then disagree else None)))
  in
  match clash with None -> `Separated | Some _ -> `Shares_cell

let separation ~name ~dims ~ranges ~concurrent ~syms a =
  let range s = find_range ranges s in
  let conc s = List.mem concurrent s ~equal:Idx.equal_symbol in
  let query = Q.separates ~dims ~range ~concurrent:conc ~syms a in
  let oracle =
    match oracle_separates ~dims ~ranges ~concurrent:conc ~syms a with
    | `Separated -> true
    | `Shares_cell -> false
  in
  Stdio.printf "%-60s query %-12s oracle %s\n" name (Bool.to_string query) (Bool.to_string oracle);
  claimf "%s: a proven separation agrees with the enumerated cells" name ((not query) || oracle)

let () =
  Stdio.printf "\n=== separates ===\n";
  let i = sym () and j = sym () and c = sym () in
  let r lo hi = (lo, hi) in
  separation ~name:"aligned vec [4i] run 4 separates i" ~dims:[| 16 |]
    ~ranges:[ (i, r 0 3) ]
    ~concurrent:[ i ] ~syms:[ i ]
    (vector ~length:4 [| aff [ (4, i) ] 0 |]);
  separation ~name:"unaligned vec [2i] run 4 does not separate i" ~dims:[| 16 |]
    ~ranges:[ (i, r 0 6) ]
    ~concurrent:[ i ] ~syms:[ i ]
    (vector ~length:4 [| aff [ (2, i) ] 0 |]);
  separation ~name:"trailing-lane vec [Sub;4c;Sub] run 4 separates c" ~dims:[| 2; 8; 1 |]
    ~ranges:[ (c, r 0 3) ]
    ~concurrent:[ c ] ~syms:[ c ]
    (vector ~length:4 [| sub; aff [ (4, c) ] 0; sub |]);
  (* Runs spill across rows when the minor extent is not a multiple of the run: [(i, 1)] writes
     [6i+4 .. 6i+7], [(i+1, 0)] writes [6i+6 .. 6i+9]. The aligned quotient alone claimed
     separation. *)
  separation ~name:"row-spilling vec [i;4j] run 4 (3x6) does not separate i, j" ~dims:[| 3; 6 |]
    ~ranges:[ (i, r 0 2); (j, r 0 1) ]
    ~concurrent:[ i; j ] ~syms:[ i; j ]
    (vector ~length:4 [| it i; aff [ (4, j) ] 0 |]);
  separation ~name:"row-aligned vec [i;4j] run 4 (3x8) separates i, j" ~dims:[| 3; 8 |]
    ~ranges:[ (i, r 0 2); (j, r 0 1) ]
    ~concurrent:[ i; j ] ~syms:[ i; j ]
    (vector ~length:4 [| it i; aff [ (4, j) ] 0 |]);
  separation ~name:"dynamic [?;j] separates j" ~dims:[| 3; 4 |]
    ~ranges:[ (j, r 0 3) ]
    ~concurrent:[ j ] ~syms:[ j ]
    (dynamic ~axis:0 [| fx 0; it j |]);
  separation ~name:"dynamic [?;j] does not separate i" ~dims:[| 3; 4 |]
    ~ranges:[ (i, r 0 2); (j, r 0 3) ]
    ~concurrent:[ i; j ] ~syms:[ i ]
    (dynamic ~axis:0 [| it i; it j |]);
  separation ~name:"flat store [Sub;c] separates c" ~dims:[| 2; 4 |]
    ~ranges:[ (c, r 0 7) ]
    ~concurrent:[ c ] ~syms:[ c ]
    (plain [| sub; it c |]);
  separation ~name:"diagonal [i;i] separates i" ~dims:[| 3; 3 |]
    ~ranges:[ (i, r 0 2) ]
    ~concurrent:[ i ] ~syms:[ i ]
    (plain [| it i; it i |])

(* {2 The box queries} *)

(* Inside the box: the flat address is inside the buffer, and so is every component of an axis that
   is not part of a flattened run (the per-axis reading the renderer relies on there). *)
let oracle_within ~dims ~ranges map =
  let flattened ax = ax > 0 && Idx.equal_axis_index map.(ax - 1) Idx.Sub_axis in
  let total = Array.fold dims ~init:1 ~f:( * ) in
  List.for_all
    (envs ranges (mentioned map))
    ~f:(fun env ->
      List.for_all (cells ~dims env (plain map)) ~f:(fun x -> x >= 0 && x < total)
      && Array.for_alli map ~f:(fun ax idx ->
          flattened ax
          || Idx.equal_axis_index idx Idx.Sub_axis
          ||
          match cells ~dims:[| dims.(ax) |] env (plain [| idx |]) with
          | [ v ] -> v >= 0 && v < dims.(ax)
          | _ -> false))

(* Covers the box: the enumeration visits every flat cell exactly once. *)
let oracle_covers ~dims ~ranges map =
  let total = Array.fold dims ~init:1 ~f:( * ) in
  let visits =
    List.concat_map (envs ranges (mentioned map)) ~f:(fun env -> cells ~dims env (plain map))
  in
  List.equal Int.equal (List.sort visits ~compare:Int.compare) (List.init total ~f:Fn.id)

let within ~name ~dims ~ranges map =
  let query = Q.within_box ~dims ~range:(find_range ranges) map in
  let oracle = oracle_within ~dims ~ranges map in
  Stdio.printf "%-60s query %-12s oracle %s\n" name (Bool.to_string query) (Bool.to_string oracle);
  claimf "%s: a proven containment agrees with the enumerated cells" name ((not query) || oracle)

let covers ~name ~dims ~ranges map =
  let query = Q.covers_box ~dims ~range:(find_range ranges) map in
  let oracle = oracle_covers ~dims ~ranges map in
  Stdio.printf "%-60s query %-12s oracle %s\n" name (Bool.to_string query) (Bool.to_string oracle);
  claimf "%s: a proven covering agrees with the enumerated cells" name ((not query) || oracle)

let () =
  Stdio.printf "\n=== within_box ===\n";
  let c = sym () and i = sym () and j = sym () in
  within ~name:"flat [Sub;c<8] inside 2x4" ~dims:[| 2; 4 |] ~ranges:[ (c, (0, 7)) ] [| sub; it c |];
  within ~name:"flat [Sub;c<9] leaves 2x4" ~dims:[| 2; 4 |] ~ranges:[ (c, (0, 8)) ] [| sub; it c |];
  within ~name:"[i;j] inside 2x4" ~dims:[| 2; 4 |]
    ~ranges:[ (i, (0, 1)); (j, (0, 3)) ]
    [| it i; it j |];
  within ~name:"[i;j+1] leaves 2x4" ~dims:[| 2; 4 |]
    ~ranges:[ (i, (0, 1)); (j, (0, 3)) ]
    [| it i; aff [ (1, j) ] 1 |];
  within ~name:"trailing [c;Sub] inside 3x2" ~dims:[| 3; 2 |]
    ~ranges:[ (c, (0, 2)) ]
    [| it c; sub |];
  Stdio.printf "\n=== covers_box ===\n";
  covers ~name:"flat [Sub;c] covers 2x4" ~dims:[| 2; 4 |] ~ranges:[ (c, (0, 7)) ] [| sub; it c |];
  covers ~name:"flat [Sub;2c] leaves holes in 2x4" ~dims:[| 2; 4 |]
    ~ranges:[ (c, (0, 3)) ]
    [| sub; aff [ (2, c) ] 0 |];
  covers ~name:"flat [Sub;i+4j] covers 2x4" ~dims:[| 2; 4 |]
    ~ranges:[ (i, (0, 3)); (j, (0, 1)) ]
    [| sub; aff [ (1, i); (4, j) ] 0 |];
  covers ~name:"trailing [c;Sub] leaves holes in 3x2" ~dims:[| 3; 2 |]
    ~ranges:[ (c, (0, 2)) ]
    [| it c; sub |];
  covers ~name:"[i;j] covers 2x4" ~dims:[| 2; 4 |]
    ~ranges:[ (i, (0, 1)); (j, (0, 3)) ]
    [| it i; it j |];
  covers ~name:"diagonal [i;i] does not cover 3x3" ~dims:[| 3; 3 |]
    ~ranges:[ (i, (0, 2)) ]
    [| it i; it i |]

(* {2 Counting}: [fiber_cardinality] reads [Sub_axis] in its IR meaning (it pins nothing and
   mentions nothing), which the enumeration of flat addresses must bear out. *)
let () =
  Stdio.printf "\n=== fiber_cardinality ===\n";
  let fiber ~name ~dims ~domain map =
    let ranges = List.map domain ~f:(fun (s, w) -> (s, (0, w - 1))) in
    let counts = Hashtbl.create (module Int) in
    List.iter
      (envs ranges (List.map domain ~f:fst))
      ~f:(fun env -> List.iter (cells ~dims env (plain map)) ~f:(Hashtbl.incr counts));
    let counts = Hashtbl.data counts in
    let query = Aff.fiber_cardinality ~domain map in
    let ok, shown =
      match query with
      | `Exact n -> (List.for_all counts ~f:(fun k -> k = n), Printf.sprintf "Exact %d" n)
      | `At_least n -> (List.for_all counts ~f:(fun k -> k >= n), Printf.sprintf "At_least %d" n)
    in
    Stdio.printf "%-60s query %s\n" name shown;
    claimf "%s: the fiber agrees with the enumerated cells" name ok
  in
  let c = sym () and d = sym () in
  fiber ~name:"flat [Sub;c] over c:8" ~dims:[| 2; 4 |] ~domain:[ (c, 8) ] [| sub; it c |];
  fiber ~name:"flat [Sub;c] over c:8, d:3 absent" ~dims:[| 2; 4 |]
    ~domain:[ (c, 8); (d, 3) ]
    [| sub; it c |];
  fiber ~name:"flat [Sub;c+d] over c:4, d:4" ~dims:[| 2; 4 |]
    ~domain:[ (c, 4); (d, 4) ]
    [| sub; aff [ (1, c); (1, d) ] 0 |];
  fiber ~name:"trailing-lane [Sub;4c;Sub] over c:4" ~dims:[| 2; 8; 1 |]
    ~domain:[ (c, 4) ]
    [| sub; aff [ (4, c) ] 0; sub |]

(* {2 Containment}: [read_covered_before] compares each prior write with the read in their views'
   common frame, so a flattened write can cover an ordinary read and vice versa. Every write here
   precedes the read and shares no loop with it; covered means every cell the read touches is a cell
   some write instance touches. *)
let () =
  Stdio.printf "\n=== read_covered_before ===\n";
  let access ?(write = true) ~loops ~path (a : acc) : unit Aff.access =
    {
      Aff.a_tn = ();
      a_map = a.map;
      a_write = write;
      a_dynamic = Option.is_some a.dyn;
      a_whole = false;
      a_vec_last = Option.is_some a.vec;
      a_vec_len = Option.value a.vec ~default:0;
      a_guarded = false;
      a_gated = false;
      a_rmw = false;
      a_val_syms = [];
      a_stmt_write = None;
      a_loops = loops;
      a_path = [ Aff.Stmt path; (if write then Aff.Write else Aff.Rhs) ];
    }
  in
  let covered ~name ~dims ~(read : acc * (Idx.symbol * (int * int)) list)
      ~(writes : (acc * (Idx.symbol * (int * int)) list) list) =
    let read_acc, read_loops = read in
    let query =
      Aff.read_covered_before ~dims
        ~read:(access ~write:false ~loops:read_loops ~path:1 read_acc)
        ~writes:(List.map writes ~f:(fun (w, loops) -> access ~loops ~path:0 w))
        ()
    in
    let written =
      List.concat_map writes ~f:(fun (w, loops) ->
          List.concat_map (envs loops (List.map loops ~f:fst)) ~f:(fun env -> cells ~dims env w))
    in
    let missing =
      List.find_map
        (envs read_loops (List.map read_loops ~f:fst))
        ~f:(fun env ->
          List.find (cells ~dims env read_acc) ~f:(fun x ->
              not (List.mem written x ~equal:Int.equal)))
    in
    let oracle = Option.is_none missing in
    let shown = match query with `Covered -> "Covered" | `Unknown _ -> "Unknown" in
    Stdio.printf "%-60s query %-12s oracle %s\n" name shown
      (if oracle then "covered" else "uncovered");
    claimf "%s: a proven coverage agrees with the enumerated cells" name
      (match query with `Covered -> oracle | `Unknown _ -> true)
  in
  let c = sym () and h = sym () and e = sym () and f = sym () in
  covered ~name:"flat write [Sub;c<8] covers read [h;e] (2x4)" ~dims:[| 2; 4 |]
    ~read:(plain [| it h; it e |], [ (h, (0, 1)); (e, (0, 3)) ])
    ~writes:[ (plain [| sub; it c |], [ (c, (0, 7)) ]) ];
  covered ~name:"flat write [Sub;c<4] leaves row 1 of read [h;e]" ~dims:[| 2; 4 |]
    ~read:(plain [| it h; it e |], [ (h, (0, 1)); (e, (0, 3)) ])
    ~writes:[ (plain [| sub; it c |], [ (c, (0, 3)) ]) ];
  covered ~name:"write [h;e] covers flat read [Sub;f<8]" ~dims:[| 2; 4 |]
    ~read:(plain [| sub; it f |], [ (f, (0, 7)) ])
    ~writes:[ (plain [| it h; it e |], [ (h, (0, 1)); (e, (0, 3)) ]) ];
  covered ~name:"flat vec write [Sub;4c] run 4 covers read [h;e]" ~dims:[| 2; 4 |]
    ~read:(plain [| it h; it e |], [ (h, (0, 1)); (e, (0, 3)) ])
    ~writes:[ (vector ~length:4 [| sub; aff [ (4, c) ] 0 |], [ (c, (0, 1)) ]) ];
  covered ~name:"trailing-lane vec [Sub;4c;Sub] covers read [h;e;0]" ~dims:[| 2; 4; 1 |]
    ~read:(plain [| it h; it e; fx 0 |], [ (h, (0, 1)); (e, (0, 3)) ])
    ~writes:[ (vector ~length:4 [| sub; aff [ (4, c) ] 0; sub |], [ (c, (0, 1)) ]) ];
  covered ~name:"trailing-lane vec [Sub;4c;Sub] c<1 leaves row 1" ~dims:[| 2; 4; 1 |]
    ~read:(plain [| it h; it e; fx 0 |], [ (h, (0, 1)); (e, (0, 3)) ])
    ~writes:[ (vector ~length:4 [| sub; aff [ (4, c) ] 0; sub |], [ (c, (0, 0)) ]) ]
