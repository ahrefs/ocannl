(* gh-494 waypoint 1-2: unit + brute-force-oracle coverage for the [Ir.Affine] query engine.

   [pair_conflict] and [covers_box] are sound-and-conservative decision procedures; on small extents
   their answers are checkable by exhaustive enumeration of the iteration boxes. Every case asserts
   soundness (a proven verdict must agree with the enumeration) and prints both the query verdict
   and the oracle classification, so precision gaps (query [Cross_thread] where the oracle found no
   cross-thread conflict) are visible and reviewable rather than silent. *)

open Base
module Idx = Ir.Indexing
module Aff = Ir.Affine

let sym = Idx.get_symbol
let aff terms offset = Idx.affine ~symbols:terms ~offset
let find_range ranges s = List.Assoc.find ranges s ~equal:Idx.equal_symbol

(* Enumerate all assignments of [syms] within [ranges] (inclusive bounds). *)
let envs ranges syms =
  List.fold syms ~init:[ [] ] ~f:(fun acc s ->
      let lo, hi = Option.value_exn (find_range ranges s) in
      List.concat_map acc ~f:(fun env -> List.init (hi - lo + 1) ~f:(fun k -> (s, lo + k) :: env)))

let eval_idx env (idx : Idx.axis_index) : int option =
  let v s = List.Assoc.find env s ~equal:Idx.equal_symbol in
  match idx with
  | Idx.Fixed_idx c -> Some c
  | Idx.Iterator s -> v s
  | Idx.Affine { symbols; offset } ->
      List.fold symbols ~init:(Some offset) ~f:(fun acc (c, s) ->
          match (acc, v s) with Some a, Some x -> Some (a + (c * x)) | _ -> None)
  | Idx.Sub_axis | Idx.Concat _ -> None

let eval_vec env (idcs : Idx.axis_index array) rank : int list option =
  List.init rank ~f:(fun p ->
      eval_idx env (if p < Array.length idcs then idcs.(p) else Idx.Fixed_idx 0))
  |> Option.all

(* Oracle for [pair_conflict]: enumerate the shared symbols once and each side's duplicated symbols
   independently; classify the conflicts (common cells) by whether all [pairs] agree. [`Opaque] when
   some component cannot be evaluated. *)
let oracle_conflict ~oracle_ranges ~dup_left ~dup_right ~pairs ~left ~right =
  let mentioned idcs =
    Array.to_list idcs
    |> List.concat_map ~f:(function
      | Idx.Iterator s -> [ s ]
      | Idx.Affine { symbols; _ } -> List.map symbols ~f:snd
      | _ -> [])
  in
  let dedup = List.dedup_and_sort ~compare:Idx.compare_symbol in
  let lsyms = dedup (mentioned left) and rsyms = dedup (mentioned right) in
  let shared = List.filter (dedup (lsyms @ rsyms)) ~f:(fun s -> not (dup_left s || dup_right s)) in
  let l_only = List.filter lsyms ~f:dup_left and r_only = List.filter rsyms ~f:dup_right in
  let rank = max (Array.length left) (Array.length right) in
  let conflicts = ref [] and opaque = ref false in
  List.iter (envs oracle_ranges shared) ~f:(fun senv ->
      List.iter (envs oracle_ranges l_only) ~f:(fun lenv ->
          List.iter (envs oracle_ranges r_only) ~f:(fun renv ->
              match (eval_vec (senv @ lenv) left rank, eval_vec (senv @ renv) right rank) with
              | Some lv, Some rv ->
                  if List.equal Int.equal lv rv then
                    conflicts := (senv @ lenv, senv @ renv) :: !conflicts
              | _ -> opaque := true)));
  if !opaque then `Opaque
  else if List.is_empty !conflicts then `Disjoint
  else if
    List.for_all !conflicts ~f:(fun (lenv, renv) ->
        List.for_all pairs ~f:(fun (p, p') ->
            match
              ( List.Assoc.find lenv p ~equal:Idx.equal_symbol,
                List.Assoc.find renv p' ~equal:Idx.equal_symbol )
            with
            | Some a, Some b -> a = b
            (* A parallel symbol absent from an access's environment takes every value: only a
               conflict-free pairing may treat it as confined. *)
            | _ -> false))
  then `Same_thread
  else `Cross_thread

let show_oracle = function
  | `Disjoint -> "disjoint"
  | `Same_thread -> "same-thread"
  | `Cross_thread -> "cross-thread"
  | `Opaque -> "opaque"

let show_verdict = function
  | Aff.Disjoint -> "Disjoint"
  | Aff.Same_thread -> "Same_thread"
  | Aff.Cross_thread _ -> "Cross_thread"

(* Soundness: a proven query verdict must not overclaim relative to the enumeration. An [`Opaque]
   oracle (some component it cannot evaluate) falsifies nothing: the query's proven verdicts remain
   sound because skipping an uninterpretable axis only enlarges the conflict set it reasons over. *)
let sound verdict oracle =
  match (verdict, oracle) with
  | _, `Opaque -> true
  | Aff.Disjoint, `Disjoint -> true
  | Aff.Disjoint, _ -> false
  | Aff.Same_thread, (`Disjoint | `Same_thread) -> true
  | Aff.Same_thread, _ -> false
  | Aff.Cross_thread _, _ -> true

let unsound_count = ref 0

(* A query answering where the oracle says it may not is the one thing this file exists to catch, so
   it fails the run rather than only landing in the printed tally -- which a `dune promote` would
   otherwise bless (gh-ocannl-601). *)
let unsound name =
  Int.incr unsound_count;
  Verdict.claim (name ^ ": query unsound against the oracle") false

let check_conflict ~name ~query_ranges ~oracle_ranges ~dup_left ~dup_right ~pairs ~left ~right =
  let range s = find_range query_ranges s in
  let verdict = Aff.pair_conflict ~range ~dup_left ~dup_right ~pairs ~left ~right in
  let oracle = oracle_conflict ~oracle_ranges ~dup_left ~dup_right ~pairs ~left ~right in
  let ok = sound verdict oracle in
  if not ok then unsound name;
  Stdio.printf "%-34s query %-12s oracle %-12s%s\n" name (show_verdict verdict) (show_oracle oracle)
    (if ok then "" else "  UNSOUND")

(* Oracle for [covers_box]: enumerate the mentioned symbols, count visits per cell. *)
let oracle_covers ~oracle_ranges ~dims idcs =
  let mentioned =
    Array.to_list idcs
    |> List.concat_map ~f:(function
      | Idx.Iterator s -> [ s ]
      | Idx.Affine { symbols; _ } -> List.map symbols ~f:snd
      | _ -> [])
    |> List.dedup_and_sort ~compare:Idx.compare_symbol
  in
  let cells = Hashtbl.create (module String) in
  let opaque = ref false in
  List.iter (envs oracle_ranges mentioned) ~f:(fun env ->
      match eval_vec env idcs (Array.length dims) with
      | Some v ->
          if List.existsi v ~f:(fun a x -> x < 0 || x >= dims.(a)) then opaque := true
          else Hashtbl.incr cells (String.concat ~sep:"," (List.map v ~f:Int.to_string))
      | None -> opaque := true);
  (not !opaque)
  && Hashtbl.length cells = Array.fold dims ~init:1 ~f:( * )
  && Hashtbl.for_all cells ~f:(fun n -> n = 1)

let check_covers ~name ~query_ranges ~oracle_ranges ~dims idcs =
  let range s = find_range query_ranges s in
  let query = Aff.covers_box ~range ~dims idcs in
  let oracle = oracle_covers ~oracle_ranges ~dims idcs in
  let ok = (not query) || oracle in
  if not ok then unsound name;
  Stdio.printf "%-34s query %-12b oracle %-12b%s\n" name query oracle
    (if ok then "" else "  UNSOUND")

let () =
  Stdio.printf "=== pair_conflict: named cases ===\n";
  let p = sym () and q = sym () and j = sym () and u = sym () and v = sym () and g = sym () in
  let i0 = sym () and j0 = sym () and s = sym () and t = sym () in
  let dup_of syms sm = List.mem syms sm ~equal:Idx.equal_symbol in
  let r3 = (0, 2) in
  (* Same-nest sites: both sides iterate the same loop symbols. *)
  let same_nest ~name ?(ranges = []) ~syms ~pairs left right =
    let ranges = if List.is_empty ranges then List.map syms ~f:(fun s -> (s, r3)) else ranges in
    check_conflict ~name ~query_ranges:ranges ~oracle_ranges:ranges ~dup_left:(dup_of syms)
      ~dup_right:(dup_of syms)
      ~pairs:(List.map pairs ~f:(fun s -> (s, s)))
      ~left ~right
  in
  same_nest ~name:"iterator agreement" ~syms:[ p; j ] ~pairs:[ p ]
    [| Idx.Iterator p; Idx.Iterator j |]
    [| Idx.Iterator p; Idx.Iterator j |];
  same_nest ~name:"stencil read p-1" ~syms:[ p ] ~pairs:[ p ] [| Idx.Iterator p |]
    [| aff [ (1, p) ] (-1) |];
  same_nest ~name:"strided disjoint 2p vs 2p+1" ~syms:[ p ] ~pairs:[ p ]
    [| aff [ (2, p) ] 0 |]
    [| aff [ (2, p) ] 1 |];
  same_nest ~name:"non-injective match p+q" ~syms:[ p; q ] ~pairs:[ p ]
    [| aff [ (1, p); (1, q) ] 0 |]
    [| aff [ (1, p); (1, q) ] 0 |];
  same_nest ~name:"mixed radix s+4t (fits)" ~syms:[ s; t ]
    ~ranges:[ (s, (0, 3)); (t, (0, 2)) ]
    ~pairs:[ s; t ]
    [| aff [ (1, s); (4, t) ] 0 |]
    [| aff [ (1, s); (4, t) ] 0 |];
  same_nest ~name:"mixed radix s+4t (overlaps)" ~syms:[ s; t ]
    ~ranges:[ (s, (0, 4)); (t, (0, 2)) ]
    ~pairs:[ s; t ]
    [| aff [ (1, s); (4, t) ] 0 |]
    [| aff [ (1, s); (4, t) ] 0 |];
  same_nest ~name:"fixed disjoint slices" ~syms:[ j ] ~pairs:[ j ]
    [| Idx.Fixed_idx 3; Idx.Iterator j |]
    [| Idx.Fixed_idx 5; Idx.Iterator j |];
  same_nest ~name:"transposed access" ~syms:[ p; q ] ~pairs:[ p; q ]
    [| Idx.Iterator p; Idx.Iterator q |]
    [| Idx.Iterator q; Idx.Iterator p |];
  same_nest ~name:"read row 0" ~syms:[ p; j ] ~pairs:[ p ]
    [| Idx.Iterator p; Idx.Iterator j |]
    [| Idx.Fixed_idx 0; Idx.Iterator j |];
  same_nest ~name:"offset loop bounds [2,5]" ~syms:[ p ]
    ~ranges:[ (p, (2, 5)) ]
    ~pairs:[ p ] [| Idx.Iterator p |] [| Idx.Iterator p |];
  (* A width-1 parallel symbol is substituted away before equation-level forcing; its thread
     equality holds by definition (single coordinate). *)
  same_nest ~name:"width-1 parallel symbol" ~syms:[ p; j ]
    ~ranges:[ (p, (0, 0)); (j, (0, 2)) ]
    ~pairs:[ p; j ]
    [| Idx.Iterator p; Idx.Iterator j |]
    [| Idx.Iterator p; Idx.Iterator j |];
  same_nest ~name:"rank padding" ~syms:[ p; j ] ~pairs:[ p ]
    [| Idx.Iterator p; Idx.Fixed_idx 0 |]
    [| Idx.Iterator p |];
  same_nest ~name:"sub_axis is opaque" ~syms:[ p ] ~pairs:[ p ]
    [| Idx.Sub_axis; Idx.Iterator p |]
    [| Idx.Sub_axis; Idx.Iterator p |];
  (* Static (shared) symbol: unknown range to the query, enumerated by the oracle. *)
  check_conflict ~name:"shared static symbol"
    ~query_ranges:[ (p, r3); (j, r3) ]
    ~oracle_ranges:[ (p, r3); (j, r3); (g, r3) ]
    ~dup_left:(dup_of [ p; j ])
    ~dup_right:(dup_of [ p; j ])
    ~pairs:[ (p, p) ]
    ~left:[| Idx.Iterator g; Idx.Iterator p |]
    ~right:[| Idx.Iterator g; Idx.Iterator p |];
  (* Cross-nest: distinct symbol sets, thread identity via pairing. *)
  check_conflict ~name:"cross-nest aligned"
    ~query_ranges:[ (i0, r3); (j0, r3); (u, r3); (v, r3) ]
    ~oracle_ranges:[ (i0, r3); (j0, r3); (u, r3); (v, r3) ]
    ~dup_left:(dup_of [ i0; u ])
    ~dup_right:(dup_of [ j0; v ])
    ~pairs:[ (i0, j0) ]
    ~left:[| Idx.Iterator i0; Idx.Iterator u |]
    ~right:[| Idx.Iterator j0; Idx.Iterator v |];
  check_conflict ~name:"cross-nest chain-position swap"
    ~query_ranges:[ (p, r3); (q, r3); (i0, r3); (j0, r3) ]
    ~oracle_ranges:[ (p, r3); (q, r3); (i0, r3); (j0, r3) ]
    ~dup_left:(dup_of [ p; q ])
    ~dup_right:(dup_of [ i0; j0 ])
    ~pairs:[ (p, i0); (q, j0) ]
    ~left:[| Idx.Iterator p; Idx.Iterator q |]
    ~right:[| Idx.Iterator j0; Idx.Iterator i0 |];

  Stdio.printf "\n=== pair_conflict: exhaustive sweep (same-nest, pairs=[p]) ===\n";
  let components =
    [
      ("F0", Idx.Fixed_idx 0);
      ("F1", Idx.Fixed_idx 1);
      ("p", Idx.Iterator p);
      ("q", Idx.Iterator q);
      ("2p", aff [ (2, p) ] 0);
      ("p+1", aff [ (1, p) ] 1);
      ("p+3q", aff [ (1, p); (3, q) ] 0);
      ("p+q", aff [ (1, p); (1, q) ] 0);
    ]
  in
  let ranges = [ (p, r3); (q, r3) ] in
  let tally = Hashtbl.create (module String) in
  List.iter components ~f:(fun (ln, lc) ->
      List.iter components ~f:(fun (rn, rc) ->
          let range s = find_range ranges s in
          let dup = dup_of [ p; q ] in
          let verdict =
            Aff.pair_conflict ~range ~dup_left:dup ~dup_right:dup
              ~pairs:[ (p, p) ]
              ~left:[| lc |] ~right:[| rc |]
          in
          let oracle =
            oracle_conflict ~oracle_ranges:ranges ~dup_left:dup ~dup_right:dup
              ~pairs:[ (p, p) ]
              ~left:[| lc |] ~right:[| rc |]
          in
          if not (sound verdict oracle) then (
            unsound (ln ^ " vs " ^ rn);
            Stdio.printf "UNSOUND: %s vs %s: query %s oracle %s\n" ln rn (show_verdict verdict)
              (show_oracle oracle));
          Hashtbl.incr tally (show_verdict verdict ^ "/" ^ show_oracle oracle)));
  Hashtbl.to_alist tally
  |> List.sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  |> List.iter ~f:(fun (k, n) -> Stdio.printf "%-28s %d\n" k n);

  Stdio.printf "\n=== covers_box ===\n";
  let a = sym () and b = sym () and i = sym () in
  let qr = [ (i, (0, 2)); (j, (0, 3)); (a, (0, 2)); (b, (0, 3)) ] in
  let cov ~name ?(query_ranges = qr) ?(oracle_ranges = qr) ~dims idcs =
    check_covers ~name ~query_ranges ~oracle_ranges ~dims idcs
  in
  cov ~name:"iterators cover box" ~dims:[| 3; 4 |] [| Idx.Iterator i; Idx.Iterator j |];
  cov ~name:"iterator wrong extent" ~dims:[| 4 |] [| Idx.Iterator i |];
  cov ~name:"mixed radix a+3b" ~dims:[| 12 |] [| aff [ (1, a); (3, b) ] 0 |];
  cov ~name:"mixed radix wrong product" ~dims:[| 13 |] [| aff [ (1, a); (3, b) ] 0 |];
  cov ~name:"repeated symbol (diagonal)" ~dims:[| 3; 3 |] [| Idx.Iterator i; Idx.Iterator i |];
  cov ~name:"offset misses zero" ~dims:[| 3 |] [| aff [ (1, i) ] 1 |];
  cov ~name:"stride gap 2i" ~dims:[| 6 |] [| aff [ (2, i) ] 0 |];
  cov ~name:"fixed on unit axis" ~dims:[| 1; 3 |] [| Idx.Fixed_idx 0; Idx.Iterator i |];
  cov ~name:"fixed on wide axis" ~dims:[| 2; 3 |] [| Idx.Fixed_idx 0; Idx.Iterator i |];
  cov ~name:"non-zero-based loop"
    ~query_ranges:[ (i, (1, 3)) ]
    ~oracle_ranges:[ (i, (1, 3)) ]
    ~dims:[| 3 |] [| Idx.Iterator i |];

  Stdio.printf "\n=== fiber_cardinality ===\n";
  (* Oracle: enumerate the domain box, count hits per image cell; [`Exact n] requires every image
     cell to be hit exactly [n] times, [`At_least n] requires at least [n]. *)
  let check_fiber ~name ~domain idcs =
    let counts = Hashtbl.create (module String) in
    let syms = List.map domain ~f:fst in
    let ranges = List.map domain ~f:(fun (s, w) -> (s, (0, w - 1))) in
    List.iter (envs ranges syms) ~f:(fun env ->
        match eval_vec env idcs (Array.length idcs) with
        | Some v -> Hashtbl.incr counts (String.concat ~sep:"," (List.map v ~f:Int.to_string))
        | None -> ());
    let counts = Hashtbl.data counts in
    let query = Aff.fiber_cardinality ~domain idcs in
    let ok, shown =
      match query with
      | `Exact n -> (List.for_all counts ~f:(fun c -> c = n), Printf.sprintf "Exact %d" n)
      | `At_least n -> (List.for_all counts ~f:(fun c -> c >= n), Printf.sprintf "At_least %d" n)
    in
    if not ok then unsound name;
    Stdio.printf "%-34s query %-12s oracle fibers %d..%d%s\n" name shown
      (Option.value ~default:0 (List.min_elt counts ~compare:Int.compare))
      (Option.value ~default:0 (List.max_elt counts ~compare:Int.compare))
      (if ok then "" else "  UNSOUND")
  in
  let f1 = sym () and f2 = sym () in
  check_fiber ~name:"absent symbol product" ~domain:[ (f1, 3); (f2, 4) ] [| Idx.Iterator f1 |];
  check_fiber ~name:"all pinned (mixed radix)"
    ~domain:[ (f1, 4); (f2, 3) ]
    [| aff [ (1, f1); (4, f2) ] 0 |];
  check_fiber ~name:"strided (image cells only)" ~domain:[ (f1, 3) ] [| aff [ (2, f1) ] 0 |];
  check_fiber ~name:"constant cell" ~domain:[ (f1, 3) ] [| Idx.Fixed_idx 0 |];
  check_fiber ~name:"non-injective f1+f2"
    ~domain:[ (f1, 3); (f2, 3) ]
    [| aff [ (1, f1); (1, f2) ] 0 |];
  (* Cancelling terms are not mentions: the map is constant in f1, whose width must count. *)
  check_fiber ~name:"cancelled term f1-f1" ~domain:[ (f1, 3) ] [| aff [ (1, f1); (-1, f1) ] 0 |];

  Stdio.printf "\n=== read_covered_before ===\n";
  (* Access records over a dummy node; only maps, loops, paths, and kind flags matter here. *)
  let acc ?(write = true) ?(whole = false) ?(vec_len = 0) ~loops ~path map =
    (* Call sites give the statement position; the terminal intra-statement component (gh-561) is
       appended here — [Write] for writes, [Rhs] for reads — as [Low_level.affine_accesses]
       would. *)
    let path =
      List.map path ~f:(fun k -> Aff.Stmt k) @ [ (if write then Aff.Write else Aff.Rhs) ]
    in
    {
      Aff.a_tn = "x";
      a_map = map;
      a_write = write;
      a_dynamic = false;
      a_whole = whole;
      a_vec_last = vec_len > 0;
      a_vec_len = vec_len;
      a_guarded = false;
      a_rmw = false;
      a_val_syms = [];
      a_stmt_write = None;
      a_loops = loops;
      a_path = path;
    }
  in
  (* Oracle: enumerate the static parameters and the read's loop box; each read cell must be written
     by some write instance visible to it — same common-loop iteration with an earlier statement
     path, or a lexicographically earlier common-loop iteration (the oracle's visibility is the true
     program order, deliberately wider than the procedure's same-iteration-only claim, so
     loop-carried coverage shows up as a precision gap, never as unsoundness). Thread symbols must
     agree wherever both sides bind them. *)
  let oracle_covered ~static_ranges ~thread ~(read : _ Aff.access) ~writes =
    let statics = List.map static_ranges ~f:fst in
    let rank =
      List.fold writes ~init:(Array.length read.Aff.a_map) ~f:(fun m (w : _ Aff.access) ->
          max m (Array.length w.Aff.a_map))
    in
    let lookup env s = List.Assoc.find_exn env s ~equal:Idx.equal_symbol in
    let opaque = ref false and all_covered = ref true in
    List.iter (envs static_ranges statics) ~f:(fun senv ->
        List.iter
          (envs read.a_loops (List.map read.a_loops ~f:fst))
          ~f:(fun renv ->
            match eval_vec (renv @ senv) read.a_map rank with
            | None -> opaque := true
            | Some cell ->
                let covered_by (w : _ Aff.access) =
                  if not w.Aff.a_write then false
                  else
                    let rec common rl wl =
                      match (rl, wl) with
                      | (s1, (lo1, hi1)) :: rt, (s2, (lo2, hi2)) :: wt
                        when Idx.equal_symbol s1 s2 && lo1 = lo2 && hi1 = hi2 ->
                          s1 :: common rt wt
                      | _ -> []
                    in
                    let cw = common read.a_loops w.a_loops in
                    List.exists
                      (envs w.a_loops (List.map w.a_loops ~f:fst))
                      ~f:(fun wenv ->
                        let thread_ok =
                          List.for_all w.a_loops ~f:(fun (s, _) ->
                              (not (thread s))
                              || (not (List.Assoc.mem renv s ~equal:Idx.equal_symbol))
                              || lookup wenv s = lookup renv s)
                        in
                        let ct_w = List.map cw ~f:(lookup wenv)
                        and ct_r = List.map cw ~f:(lookup renv) in
                        let cmp = List.compare Int.compare ct_w ct_r in
                        thread_ok
                        && (cmp < 0
                           || cmp = 0
                              && List.compare Aff.compare_path_comp w.a_path read.a_path < 0)
                        &&
                        if w.a_whole then true
                        else
                          match eval_vec (wenv @ senv) w.a_map rank with
                          | None -> false
                          | Some wcell ->
                              if w.a_vec_last then
                                let va = Array.length w.a_map - 1 in
                                List.for_alli cell ~f:(fun p c ->
                                    let wc = List.nth_exn wcell p in
                                    if p = va then c >= wc && c < wc + w.a_vec_len else c = wc)
                              else List.equal Int.equal wcell cell)
                in
                if not (List.exists writes ~f:covered_by) then all_covered := false));
    if !opaque then `Opaque else if !all_covered then `Covered else `Uncovered
  in
  let check_containment ~name ?(static_ranges = []) ?(thread = fun _ -> false) ~read ~writes () =
    let static_range s = find_range static_ranges s in
    let query = Aff.read_covered_before ~thread ~static_range ~read ~writes () in
    let oracle = oracle_covered ~static_ranges ~thread ~read ~writes in
    let qs = match query with `Covered -> "Covered" | `Unknown _ -> "Unknown" in
    let os =
      match oracle with `Covered -> "covered" | `Uncovered -> "uncovered" | `Opaque -> "opaque"
    in
    let ok = match (query, oracle) with `Covered, `Uncovered -> false | _ -> true in
    if not ok then unsound name;
    Stdio.printf "%-34s query %-12s oracle %-12s%s\n" name qs os (if ok then "" else "  UNSOUND")
  in
  let i1 = sym () and k1 = sym () and u1 = sym () and st = sym () in
  let l8 s = (s, (0, 7)) in
  (* Sequential statements: full overwrite then read. *)
  check_containment ~name:"overwrite covers read"
    ~read:(acc ~write:false ~loops:[ l8 u1 ] ~path:[ 1 ] [| Idx.Iterator u1 |])
    ~writes:[ acc ~loops:[ l8 i1 ] ~path:[ 0 ] [| Idx.Iterator i1 |] ]
    ();
  check_containment ~name:"read wider than write"
    ~read:(acc ~write:false ~loops:[ (u1, (0, 8)) ] ~path:[ 1 ] [| Idx.Iterator u1 |])
    ~writes:[ acc ~loops:[ l8 i1 ] ~path:[ 0 ] [| Idx.Iterator i1 |] ]
    ();
  check_containment ~name:"strided write, strided read"
    ~read:(acc ~write:false ~loops:[ (u1, (0, 3)) ] ~path:[ 1 ] [| aff [ (2, u1) ] 0 |])
    ~writes:[ acc ~loops:[ (i1, (0, 3)) ] ~path:[ 0 ] [| aff [ (2, i1) ] 0 |] ]
    ();
  check_containment ~name:"strided write, dense read"
    ~read:(acc ~write:false ~loops:[ l8 u1 ] ~path:[ 1 ] [| Idx.Iterator u1 |])
    ~writes:[ acc ~loops:[ (i1, (0, 3)) ] ~path:[ 0 ] [| aff [ (2, i1) ] 0 |] ]
    ();
  (* Union along one axis: an init cell plus a shifted loop write jointly cover. *)
  check_containment ~name:"union init + shifted write"
    ~read:(acc ~write:false ~loops:[ l8 u1 ] ~path:[ 2 ] [| Idx.Iterator u1 |])
    ~writes:
      [
        acc ~loops:[] ~path:[ 0 ] [| Idx.Fixed_idx 0 |];
        acc ~loops:[ (i1, (0, 6)) ] ~path:[ 1 ] [| aff [ (1, i1) ] 1 |];
      ]
    ();
  check_containment ~name:"union with gap"
    ~read:(acc ~write:false ~loops:[ l8 u1 ] ~path:[ 2 ] [| Idx.Iterator u1 |])
    ~writes:
      [
        acc ~loops:[] ~path:[ 0 ] [| Idx.Fixed_idx 0 |];
        acc ~loops:[ (i1, (0, 5)) ] ~path:[ 1 ] [| aff [ (1, i1) ] 2 |];
      ]
    ();
  (* Same statement (equal path): never a prior write. *)
  check_containment ~name:"same-statement write unusable"
    ~read:(acc ~write:false ~loops:[ l8 u1 ] ~path:[ 0 ] [| Idx.Iterator u1 |])
    ~writes:[ acc ~loops:[ l8 u1 ] ~path:[ 0 ] [| Idx.Iterator u1 |] ]
    ();
  (* Common enclosing loop cancels on both sides. *)
  check_containment ~name:"common loop cancels"
    ~read:
      (acc ~write:false
         ~loops:[ (i1, (0, 2)); (u1, (0, 2)) ]
         ~path:[ 0; 1 ]
         [| Idx.Iterator i1; Idx.Iterator u1 |])
    ~writes:
      [
        acc
          ~loops:[ (i1, (0, 2)); (k1, (0, 3)) ]
          ~path:[ 0; 0 ]
          [| Idx.Iterator i1; Idx.Iterator k1 |];
      ]
    ();
  (* Write-side residual common-loop symbol: declined (future iterations), though the true program
     order covers it — a documented precision gap. *)
  check_containment ~name:"write-side residual common sym"
    ~read:(acc ~write:false ~loops:[ (i1, (0, 2)) ] ~path:[ 0; 1 ] [| Idx.Fixed_idx 0 |])
    ~writes:[ acc ~loops:[ (i1, (0, 2)) ] ~path:[ 0; 0 ] [| Idx.Iterator i1 |] ]
    ();
  (* Loop-carried coverage (read x[i-1] on the rhs of the statement writing x[i], after an x[0] init
     statement): declined, oracle covers. gh-561 note: this used to hand-build the shift statement's
     read and write at sibling positions and the init at their PREFIX — the ambiguity the
     intra-statement components remove; now the read/write share the shift statement's position,
     ordered [Rhs] before [Write], and the init is an honestly earlier statement. *)
  check_containment ~name:"loop-carried shift declined"
    ~read:(acc ~write:false ~loops:[ (i1, (1, 3)) ] ~path:[ 1 ] [| aff [ (1, i1) ] (-1) |])
    ~writes:
      [
        acc ~loops:[] ~path:[ 0 ] [| Idx.Fixed_idx 0 |];
        acc ~loops:[ (i1, (1, 3)) ] ~path:[ 1 ] [| Idx.Iterator i1 |];
      ]
    ();
  (* Whole-node write (Zero_out). *)
  check_containment ~name:"whole-node write covers"
    ~read:(acc ~write:false ~loops:[ l8 u1 ] ~path:[ 1 ] [| Idx.Iterator u1; Idx.Iterator u1 |])
    ~writes:[ acc ~whole:true ~loops:[] ~path:[ 0 ] [||] ]
    ();
  (* Vectorized write: strided bases with abutting runs are dense. *)
  check_containment ~name:"vec write abutting runs"
    ~read:(acc ~write:false ~loops:[ l8 u1 ] ~path:[ 1 ] [| Idx.Iterator u1 |])
    ~writes:[ acc ~vec_len:4 ~loops:[ (i1, (0, 1)) ] ~path:[ 0 ] [| aff [ (4, i1) ] 0 |] ]
    ();
  (* Thread-identity symbols: per-thread coverage. *)
  let thr = List.mem [ i1 ] ~equal:Idx.equal_symbol in
  check_containment ~name:"thread cancels (own cells)" ~thread:thr
    ~read:
      (acc ~write:false
         ~loops:[ (i1, (0, 2)); (u1, (0, 3)) ]
         ~path:[ 0; 1 ]
         [| Idx.Iterator i1; Idx.Iterator u1 |])
    ~writes:
      [
        acc
          ~loops:[ (i1, (0, 2)); (k1, (0, 3)) ]
          ~path:[ 0; 0 ]
          [| Idx.Iterator i1; Idx.Iterator k1 |];
      ]
    ();
  check_containment ~name:"thread reads foreign cell" ~thread:thr
    ~read:(acc ~write:false ~loops:[ (i1, (0, 2)) ] ~path:[ 0; 1 ] [| Idx.Iterator i1 |])
    ~writes:[ acc ~loops:[ (i1, (0, 2)) ] ~path:[ 0; 0 ] [| Idx.Fixed_idx 0 |] ]
    ();
  (* A write not under the thread loop executes redundantly on every thread (bare statement): each
     thread's copy receives all its cells. *)
  check_containment ~name:"thread read over bare write" ~thread:thr
    ~read:(acc ~write:false ~loops:[ (i1, (0, 2)) ] ~path:[ 1 ] [| Idx.Iterator i1 |])
    ~writes:[ acc ~loops:[ (k1, (0, 3)) ] ~path:[ 0 ] [| Idx.Iterator k1 |] ]
    ();
  (* Static parameters: cancelled when matched, universalized on the read side when ranged. *)
  check_containment ~name:"static cancels"
    ~static_ranges:[ (st, (0, 5)) ]
    ~read:(acc ~write:false ~loops:[] ~path:[ 1 ] [| Idx.Iterator st |])
    ~writes:[ acc ~loops:[] ~path:[ 0 ] [| Idx.Iterator st |] ]
    ();
  check_containment ~name:"static read over full write"
    ~static_ranges:[ (st, (0, 5)) ]
    ~read:(acc ~write:false ~loops:[] ~path:[ 1 ] [| Idx.Iterator st |])
    ~writes:[ acc ~loops:[ l8 i1 ] ~path:[ 0 ] [| Idx.Iterator i1 |] ]
    ();
  check_containment ~name:"static read over fixed write"
    ~static_ranges:[ (st, (0, 5)) ]
    ~read:(acc ~write:false ~loops:[] ~path:[ 1 ] [| Idx.Iterator st |])
    ~writes:[ acc ~loops:[] ~path:[ 0 ] [| Idx.Fixed_idx 0 |] ]
    ();
  (* Multi-symbol dense vs gapped write images. *)
  check_containment ~name:"mixed-radix write dense"
    ~read:(acc ~write:false ~loops:[ l8 u1 ] ~path:[ 1 ] [| Idx.Iterator u1 |])
    ~writes:
      [ acc ~loops:[ (i1, (0, 1)); (k1, (0, 3)) ] ~path:[ 0 ] [| aff [ (4, i1); (1, k1) ] 0 |] ]
    ();
  check_containment ~name:"mixed-radix write gapped"
    ~read:(acc ~write:false ~loops:[ (u1, (0, 6)) ] ~path:[ 1 ] [| Idx.Iterator u1 |])
    ~writes:
      [ acc ~loops:[ (i1, (0, 1)); (k1, (0, 2)) ] ~path:[ 0 ] [| aff [ (4, i1); (1, k1) ] 0 |] ]
    ();
  (* Multi-axis box union: padded-margin strips plus the interior tile the read box. *)
  let v1 = sym () in
  check_containment ~name:"2-D margins + interior tile"
    ~read:
      (acc ~write:false
         ~loops:[ (u1, (0, 4)); (v1, (0, 4)) ]
         ~path:[ 5 ]
         [| Idx.Iterator u1; Idx.Iterator v1 |])
    ~writes:
      [
        acc ~loops:[ (i1, (0, 5)) ] ~path:[ 0 ] [| Idx.Fixed_idx 0; Idx.Iterator i1 |];
        acc ~loops:[ (i1, (1, 3)) ] ~path:[ 1 ] [| Idx.Iterator i1; Idx.Fixed_idx 0 |];
        acc ~loops:[ (i1, (1, 3)); (k1, (4, 5)) ] ~path:[ 2 ] [| Idx.Iterator i1; Idx.Iterator k1 |];
        acc ~loops:[ (i1, (4, 5)); (k1, (0, 5)) ] ~path:[ 3 ] [| Idx.Iterator i1; Idx.Iterator k1 |];
        acc
          ~loops:[ (i1, (0, 2)); (k1, (0, 2)) ]
          ~path:[ 4 ]
          [| aff [ (1, i1) ] 1; aff [ (1, k1) ] 1 |];
      ]
    ();
  check_containment ~name:"2-D margins with hole"
    ~read:
      (acc ~write:false
         ~loops:[ (u1, (0, 4)); (v1, (0, 4)) ]
         ~path:[ 5 ]
         [| Idx.Iterator u1; Idx.Iterator v1 |])
    ~writes:
      [
        acc ~loops:[ (i1, (0, 5)) ] ~path:[ 0 ] [| Idx.Fixed_idx 0; Idx.Iterator i1 |];
        acc ~loops:[ (i1, (1, 3)) ] ~path:[ 1 ] [| Idx.Iterator i1; Idx.Fixed_idx 0 |];
        acc ~loops:[ (i1, (4, 5)); (k1, (0, 5)) ] ~path:[ 2 ] [| Idx.Iterator i1; Idx.Iterator k1 |];
        acc
          ~loops:[ (i1, (0, 2)); (k1, (0, 2)) ]
          ~path:[ 3 ]
          [| aff [ (1, i1) ] 1; aff [ (1, k1) ] 1 |];
      ]
    ();
  (* Pointwise literal initialization tiles the box. *)
  check_containment ~name:"pointwise writes tile the box"
    ~read:
      (acc ~write:false
         ~loops:[ (u1, (0, 1)); (v1, (0, 1)) ]
         ~path:[ 4 ]
         [| Idx.Iterator u1; Idx.Iterator v1 |])
    ~writes:
      (List.concat_map [ 0; 1 ] ~f:(fun a ->
           List.map [ 0; 1 ] ~f:(fun b ->
               acc ~loops:[] ~path:[ (2 * a) + b ] [| Idx.Fixed_idx a; Idx.Fixed_idx b |])))
    ();

  Stdio.printf "\n=== gh-561: intra-statement path components ===\n";
  (* The trap shape (gh-554 round 3): [if (E[i] < 1) then E[i] = ...] — the condition's read and the
     guarded body's write used to share one statement position, so a consumer ordering or matching
     reads against writes by path aliased them. The encoding now keeps them apart. *)
  let cond_read = [ Aff.Stmt 3; Aff.Cond ] in
  let body_write = [ Aff.Stmt 3; Aff.Body; Aff.Write ] in
  let rhs_read = [ Aff.Stmt 3; Aff.Body; Aff.Rhs ] in
  let p name b =
    Stdio.printf "%-64s %b\n" name b;
    Verdict.claim name b
  in
  p "cond read is NOT statement-subordinate to the guarded body's write"
    (not (Aff.same_statement cond_read body_write));
  p "the body's own rhs read IS statement-subordinate to its write"
    (Aff.same_statement rhs_read body_write);
  p "program order: cond read before the guarded body's write"
    (List.compare Aff.compare_path_comp cond_read body_write < 0);
  p "program order: a statement's rhs before its own write"
    (List.compare Aff.compare_path_comp rhs_read body_write < 0);
  (* The [Local_scope] case that used to need [read_covered_before]'s prefix-exclusion rule: the
     enclosing statement's write now orders after reads in its inlined rhs body by the [Rhs] <
     [Write] component, so it cannot pose as a covering prior write. *)
  let scope_body_read = [ Aff.Stmt 3; Aff.Rhs; Aff.Stmt 0; Aff.Rhs ] in
  let enclosing_write = [ Aff.Stmt 3; Aff.Write ] in
  p "program order: an inlined scope body's read before the enclosing write"
    (List.compare Aff.compare_path_comp scope_body_read enclosing_write < 0)

(* gh-ocannl-722 / gh-ocannl-721: the peel-guard legality queries.

   [separates] is the instance-vs-instance form of [pair_conflict] — one access taken twice — so it
   is checked against the same enumeration oracle, and its soundness direction is the same: an
   answer of "separated" must agree with the enumeration, while a conservative "not separated" over
   a cell the oracle finds injective is a precision gap and is printed rather than failed.

   [peel_guard] is pure symbol-set classification, so it has no oracle: what its cases assert is
   which of the three answers each shape earns, the third being the one that hands the question on
   to [separates]. *)
let () =
  Stdio.printf "\n=== gh-722: separates (the peel's lane-private escape) ===\n";
  let w = sym () and w2 = sym () and k = sym () and s = sym () in
  let ranges = [ (w, (0, 3)); (w2, (0, 3)); (k, (0, 3)) ] in
  (* [s] is a STATIC symbol: the query has no range for it (it is bound outside every loop, so
     [range] answers [None] and the engine keeps it shared between the two instances), while the
     enumeration still has to give it values. *)
  let oracle_ranges = (s, (0, 1)) :: ranges in
  let concurrent_of syms sm = List.mem syms sm ~equal:Idx.equal_symbol in
  let check_separates ~name ~concurrent ~syms idcs =
    let range x = find_range ranges x in
    let conc = concurrent_of concurrent in
    let query = Aff.separates ~range ~concurrent:conc ~syms ~idcs in
    let oracle =
      match
        oracle_conflict ~oracle_ranges ~dup_left:conc ~dup_right:conc
          ~pairs:(List.map syms ~f:(fun x -> (x, x)))
          ~left:idcs ~right:idcs
      with
      | `Disjoint | `Same_thread -> true
      | `Cross_thread -> false
      | `Opaque -> true
    in
    let ok = (not query) || oracle in
    if not ok then unsound name;
    Stdio.printf "%-46s query %-6b oracle %-6b%s\n" name query oracle
      (if ok then "" else "  UNSOUND")
  in
  (* The shape gh-ocannl-721 is about: [acc[w]] under [Workgroup w], every lane its own cell. *)
  check_separates ~name:"acc[w] separates w" ~concurrent:[ w ] ~syms:[ w ] [| Idx.Iterator w |];
  (* The round-3 race: one cell for every lane. *)
  check_separates ~name:"acc[0] does not separate w" ~concurrent:[ w ] ~syms:[ w ]
    [| Idx.Fixed_idx 0 |];
  (* Mentioning the symbol is not separating it: [(0,1)] and [(1,0)] address the same cell. *)
  check_separates ~name:"acc[w + w2] separates neither" ~concurrent:[ w; w2 ] ~syms:[ w ]
    [| aff [ (1, w); (1, w2) ] 0 |];
  (* A mixed-radix cell does separate both, and that is exactly the criterion [forced_pairs]
     uses. *)
  check_separates ~name:"acc[4w + w2] separates w and w2" ~concurrent:[ w; w2 ] ~syms:[ w; w2 ]
    [| aff [ (4, w); (1, w2) ] 0 |];
  (* Two axes, one per lane symbol. *)
  check_separates ~name:"acc[w, w2] separates both" ~concurrent:[ w; w2 ] ~syms:[ w; w2 ]
    [| Idx.Iterator w; Idx.Iterator w2 |];
  (* A static offset is shared between the two instances, so it cancels rather than blurring. *)
  check_separates ~name:"acc[w + s] separates w (s static)" ~concurrent:[ w ] ~syms:[ w ]
    [| aff [ (1, w); (1, s) ] 0 |];
  (* Nothing to tell apart is separated vacuously — the peel's "no enclosing symbol" case. *)
  check_separates ~name:"an empty symbol set is separated" ~concurrent:[ w ] ~syms:[]
    [| Idx.Fixed_idx 0 |];
  (* An index component the engine cannot interpret contributes no information, so the answer is the
     conservative one rather than an unsound "separated". *)
  check_separates ~name:"an opaque component does not separate" ~concurrent:[ w ] ~syms:[ w ]
    [| Idx.Sub_axis |];

  (* {!Aff.within_box}: access validity, the half separation does not answer (Codex P1 on PR #443).
     No oracle -- an interval containment over a box is decided by the same arithmetic an
     enumeration would perform -- so these are named cases, each naming which way it must go. *)
  Stdio.printf "\n=== gh-722: within_box (the validity half) ===\n";
  let wb name ~dims idcs expected =
    let got = Aff.within_box ~range:(fun x -> find_range ranges x) ~dims idcs in
    Stdio.printf "%-52s %b (want %b)\n" name got expected;
    Verdict.claim name (Bool.equal got expected)
  in
  wb "acc[w] fits a 4-cell node" ~dims:[| 4 |] [| Idx.Iterator w |] true;
  wb "acc[w] does not fit a 1-cell node" ~dims:[| 1 |] [| Idx.Iterator w |] false;
  wb "acc[w] does not fit a 3-cell node either" ~dims:[| 3 |] [| Idx.Iterator w |] false;
  wb "acc[w + 1] leaves a 4-cell node at the top" ~dims:[| 4 |] [| aff [ (1, w) ] 1 |] false;
  wb "acc[w - 1] leaves a 4-cell node at the bottom" ~dims:[| 4 |] [| aff [ (1, w) ] (-1) |] false;
  wb "acc[4w + w2] fits a 16-cell node" ~dims:[| 16 |] [| aff [ (4, w); (1, w2) ] 0 |] true;
  wb "acc[w, w2] fits a 4x4 node" ~dims:[| 4; 4 |] [| Idx.Iterator w; Idx.Iterator w2 |] true;
  wb "acc[w, w2] does not fit a 4x2 node" ~dims:[| 4; 2 |]
    [| Idx.Iterator w; Idx.Iterator w2 |]
    false;
  wb "a fixed index inside the box fits" ~dims:[| 4 |] [| Idx.Fixed_idx 2 |] true;
  wb "a fixed index outside it does not" ~dims:[| 2 |] [| Idx.Fixed_idx 2 |] false;
  (* A static parameter's value is not known here, so it cannot be placed inside any box -- the
     conservative answer, this query being asked only to license moving an access out from under a
     guard. *)
  wb "a static index parameter is not placed" ~dims:[| 4 |] [| Idx.Iterator s |] false;
  wb "an opaque component is not placed" ~dims:[| 4 |] [| Idx.Sub_axis |] false;
  wb "a rank mismatch is not placed" ~dims:[| 4; 4 |] [| Idx.Iterator w |] false;

  Stdio.printf "\n=== gh-722: peel_guard (which guards may join the peeled levels) ===\n";
  let p name b =
    Stdio.printf "%-72s %b\n" name b;
    Verdict.claim name b
  in
  let loop_bound x = List.mem [ w; w2; k ] x ~equal:Idx.equal_symbol in
  let peeled x = Idx.equal_symbol x k in
  let verdict guard_syms = Aff.peel_guard ~loop_bound ~peeled ~guard_syms in
  let is_confined g = match verdict g with Aff.Confined_to_peel -> true | _ -> false in
  let is_rejected g = match verdict g with Aff.Not_peelable _ -> true | _ -> false in
  let pending g =
    match verdict g with Aff.Lane_private_if_separated syms -> Some syms | _ -> None
  in
  p "a guard over the peeled index alone is confined" (is_confined [ k ]);
  p "a guard over the peeled index and a static bound is confined" (is_confined [ k; s ]);
  p "a guard mentioning no index at all is refused" (is_rejected []);
  p "a guard over an enclosing index alone is refused" (is_rejected [ w ]);
  p "a guard over a static bound alone is refused" (is_rejected [ s ]);
  p "a mixed guard defers to the cell, naming the enclosing symbol"
    (match pending [ w; k ] with Some [ x ] -> Idx.equal_symbol x w | _ -> false);
  p "a mixed guard names every enclosing symbol it mentions, once"
    (match pending [ w; k; w2; w ] with
    | Some syms ->
        List.length syms = 2
        && List.mem syms w ~equal:Idx.equal_symbol
        && List.mem syms w2 ~equal:Idx.equal_symbol
    | _ -> false);
  p "a mixed guard's static bound is not something the cell must separate"
    (match pending [ w; k; s ] with Some [ x ] -> Idx.equal_symbol x w | _ -> false)

(* The tally, last: every section above has run by now, so a case any of them found unsound is
   counted here as well as failing its own claim. *)
let () = Stdio.printf "\nunsound cases: %d\n" !unsound_count
