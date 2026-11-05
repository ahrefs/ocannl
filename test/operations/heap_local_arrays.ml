(* gh-ocannl-1669: the cc backend declares a [Local]-placement array on the stack only within a byte
   budget, and past it on the heap -- where the arrays had gone on the stack at any size, bounded
   only by the 8 MiB main-thread stack (the whole-B~ pack of a Grid-outermost GEBP was held under 4
   MiB for exactly that reason).

   Two scopes, two budgets ([Ir.C_syntax.routine_stack_locals_bytes],
   [Ir.C_syntax.chunk_stack_locals_bytes], both read here rather than restated): the routine's
   function-scope arrays, and a pool-parallel [Grid] chunk's privatized tiles, which live on a pool
   worker's stack -- 512 KiB under libdispatch. Each scope is pinned on both sides of its budget, in
   the emitted source and by executing it:

   - [at]: one array exactly at the budget is a stack array; - [over]: one element more is a heap
   array, freed in its scope; - [sum]: the budget bounds the scope's TOTAL, smallest array first --
   two arrays each within it but together over it put the smaller on the stack and the larger on the
   heap; - [big]: a 12 MiB array, past even the main thread's 8 MiB stack. Before the heap
   declarations, the function-scope one died on the main thread and the privatized one on every pool
   worker (it needs the raised privatization cap below to be privatized at all).

   Every array is filled forward with values that identify the iteration and drained in REVERSE into
   a materialized output (a forward drain lets the C compiler forward every store and drop the
   array), so each leg's executed values check the whole array round trip. *)

open Base
open Ll_test
open Verdict.Claims
module Generated = Test_utils.Generated

let backend_name = "cc"
let () = Utils.settings.output_debug_files_in_build_directory <- true
let () = Generated.init ~backend_name

(* gh-ocannl-1778: the [outlined] mode is this test run with every kernel forced outlined
   ([cc_kernel_part_weight=1], [cc_compile_unit_min_bytes=0] on its rule), under the plain run's
   golden. A Grid chunk's privatized tile is declared inside the loop, so an outlined kernel
   declares it in the part its loop moved to, while a function-scope array stays in the kernel's own
   frame: the mode claims each Grid tile was read from a part, which a declarations reading that
   misses the parts cannot satisfy, and which a threshold change leaving the kernels whole fails
   too. *)
let outlined =
  match
    List.filter
      (List.tl_exn (Array.to_list Stdlib.Sys.argv))
      ~f:(fun arg -> not (String.is_prefix arg ~prefix:"--"))
  with
  | [] -> false
  | [ "outlined" ] -> true
  | _ -> failwith "usage: heap_local_arrays.exe [outlined] [--ocannl_<key>=<value>...]"

(* The Grid legs pin the stack budget of a privatized tile, so they need tiles the privatization
   rule admits past it: [cc_grid_private_bytes_cap] (a lazy the first compile forces, stated here
   before any) raised over the 12 MiB tile; and every launch admitted whatever its work estimate. *)
let () =
  Hashtbl.set Utils.config_file_args ~key:"cc_grid_private_bytes_cap" ~data:"67108864";
  Hashtbl.set Utils.config_file_args ~key:"cc_grid_min_updates" ~data:"0"

let ctx = Context.cpu ()
let mk = node_factory ~first_id:16690 ~dims:[| 1 |] ()
let floats bytes = bytes / 4
let big_elems = floats (12 * 1024 * 1024)
let grid_extent = 2

let scratch label n =
  let tn = mk ~dims:[| n |] label in
  Tn.update_memory_mode tn Tn.Local (Site "99:test-setup");
  tn

let rev ~n x = aff [ (-1, x) ] (n - 1)

(* The function-scope legs: each tile, of its own extent [n_t], holds [1 + x] at [x]; the output, of
   the smallest tile's extent [m], sums at [y] every tile's cell [n_t - 1 - y]. *)
let routine ~name tiles =
  let extent t = (Lazy.force t.Tn.dims).(0) in
  let m = List.fold tiles ~init:Int.max_value ~f:(fun acc t -> min acc (extent t)) in
  let out = mk ~dims:[| m |] (name ^ "_out") in
  materialize out;
  let fills =
    List.map tiles ~f:(fun t ->
        let x = sym () in
        loop_n x (extent t) (set t [| iter x |] (tick x)))
  in
  let y = sym () in
  let drain =
    loop_n y m
      (set out
         [| iter y |]
         (List.fold tiles ~init:(c 0.) ~f:(fun acc t -> add acc (get t [| rev ~n:(extent t) y |]))))
  in
  let llc = List.fold_right fills ~init:drain ~f:seq in
  let want =
    Array.init m ~f:(fun y ->
        Float.of_int (List.sum (module Int) tiles ~f:(fun t -> 1 + (extent t - 1 - y))))
  in
  (optimize ~materialized:[ out ] ~name llc, out, want)

(* The Grid legs: under a [Grid] loop over [g], the privatized tile holds [1 + 10 g + x] at [x],
   rewritten wholly by each iteration before it is read; the output at [g, x] is its cell [n - 1 -
   x]. A short loop filling a second output comes first: a kernel of one top-level statement stays
   whole at any part weight ([Ir.Kernel_outline.groups]), and the [outlined] mode needs the Grid
   loop in a part of its own. *)
let grid_routine ~name tile ~n =
  let out = mk ~dims:[| grid_extent; n |] (name ^ "_out") in
  let lead = mk ~dims:[| grid_extent |] (name ^ "_lead") in
  materialize out;
  materialize lead;
  let g = sym () and x = sym () and y = sym () and z = sym () in
  let llc =
    seq
      (loop_n z grid_extent (set lead [| iter z |] (tick z)))
      (loop_n ~axis:LL.Grid g grid_extent
         (seq
            (loop_n x n (set tile [| iter x |] (tag g x)))
            (loop_n y n (set out [| iter g; iter y |] (get tile [| rev ~n y |])))))
  in
  let want =
    Array.init (grid_extent * n) ~f:(fun cell ->
        let g = cell / n and x = cell % n in
        Float.of_int (1 + (10 * g) + (n - 1 - x)))
  in
  (optimize ~materialized:[ out ] ~name llc, out, want)

(* The emitted forms of a tile, by its identifier (its label: each is unique in its routine and
   clear of every reserved name; every tile is single precision), read structurally
   ([Generated.declared_arrays]): a stack array declares [float ident[n]], a heap array a [float *]
   named [ident] into the allocation [free] takes, named by the renderer's own derivation -- each in
   the scope claimed. Each form is claimed present, never only the other absent, so an identifier
   spelled otherwise fails rather than passes. *)
let allocation ident = Ir.C_syntax.derived_name Ir.C_syntax.heap_local_derivation ident

let declarations src ~ident =
  List.filter (Generated.declared_arrays src) ~f:(fun a -> String.equal a.Generated.ident ident)

let in_scope ~scope (a : Generated.declared_array) =
  match (scope, a.scope) with
  | `Function, Function_scope | `Block, Block_scope _ -> true
  | _ -> false

let on_stack src ~scope ~ident ~n =
  match declarations src ~ident with
  | [ a ] -> (not a.heap) && a.elems = n && String.equal a.typ "float" && in_scope ~scope a
  | _ -> false

let on_heap src ~scope ~ident ~n =
  (match declarations src ~ident with
    | [ a ] -> a.heap && a.elems = n && String.equal a.typ "float" && in_scope ~scope a
    | _ -> false)
  && String.is_substring src ~substring:(Printf.sprintf "free(%s);" (allocation ident))

(* Where the [outlined] mode claims a Grid tile was declared: inside an outlined part. *)
let in_part src ~ident =
  match declarations src ~ident with
  | [ { scope = Block_scope { part = Some _ }; _ } ] -> true
  | _ -> false

let pool_rendered src = String.is_substring src ~substring:"Pool-backed Grid rendering"

let run_leg ~name (o, out, want) =
  let got = List.hd_exn (execute ~ctx ~name o ~seed:[] ~read:[ out ]) in
  (Generated.read name, got, want)

let routine_budget = Ir.C_syntax.routine_stack_locals_bytes

(* [tiles] are [(label, bytes)]; [heap] the labels expected on the heap, the rest on the stack. *)
let function_leg ~what ~name tiles ~heap =
  let tns = List.map tiles ~f:(fun (label, bytes) -> (label, scratch label (floats bytes))) in
  let src, got, want = run_leg ~name (routine ~name (List.map tns ~f:snd)) in
  List.iter tns ~f:(fun (label, tn) ->
      if List.mem heap label ~equal:String.equal then
        pf "function scope, %s: %s is a heap array, freed" what label
          (on_heap src ~scope:`Function ~ident:label ~n:(Tn.num_elems tn))
      else
        pf "function scope, %s: %s is a stack array" what label
          (on_stack src ~scope:`Function ~ident:label ~n:(Tn.num_elems tn)));
  pf_all2 "function scope, %s: values" what got want ~f:Float.equal

let () =
  function_leg ~what:"at the budget" ~name:"hla_fs_at" [ ("hla_fs_at", routine_budget) ] ~heap:[];
  function_leg ~what:"one element over the budget" ~name:"hla_fs_over"
    [ ("hla_fs_over", routine_budget + 4) ]
    ~heap:[ "hla_fs_over" ];
  (* The larger declared first, so a decision in declaration order would keep it instead. *)
  function_leg ~what:"two arrays over the budget together" ~name:"hla_fs_sum"
    [ ("hla_fs_large", routine_budget * 3 / 4); ("hla_fs_small", routine_budget / 2) ]
    ~heap:[ "hla_fs_large" ];
  function_leg ~what:"12 MiB" ~name:"hla_fs_big"
    [ ("hla_fs_big", 4 * big_elems) ]
    ~heap:[ "hla_fs_big" ]

(* A node labeled like a name the heap declarations call (local review of the PR): the renderer
   reserves those names, so the node is declared under another and the library's [free] stays
   visible -- otherwise the kernel calls the node and does not compile. *)
let () =
  let name = "hla_fs_named" in
  let heap_tile = scratch name (floats routine_budget + 1) and named = scratch "free" 16 in
  let src, got, want = run_leg ~name (routine ~name [ heap_tile; named ]) in
  p "function scope, beside a node labeled free: the heap array, freed"
    (on_heap src ~scope:`Function ~ident:name ~n:(Tn.num_elems heap_tile));
  p_none "function scope, beside a node labeled free: no array is declared as free"
    (Generated.declared_arrays src) ~f:(fun a -> String.equal a.ident "free");
  p_all2 "function scope, beside a node labeled free: values" got want ~f:Float.equal

let grid_leg ~what ~name ~n ~heap =
  let tile = scratch (name ^ "_tile") n in
  let src, got, want = run_leg ~name (grid_routine ~name tile ~n) in
  let id = name ^ "_tile" in
  pf "Grid chunk, %s: the Grid renders on the pool" what (pool_rendered src);
  if heap then pf "Grid chunk, %s: a heap tile, freed" what (on_heap src ~scope:`Block ~ident:id ~n)
  else pf "Grid chunk, %s: a stack tile" what (on_stack src ~scope:`Block ~ident:id ~n);
  if outlined then
    Verdict.claimf "Grid chunk, %s: the forced run declared the tile inside an outlined part" what
      (in_part src ~ident:id);
  pf_all2 "Grid chunk, %s: values" what got want ~f:Float.equal

let () =
  let n = floats Ir.C_syntax.chunk_stack_locals_bytes in
  grid_leg ~what:"at the budget" ~name:"hla_gr_at" ~n ~heap:false;
  grid_leg ~what:"one element over the budget" ~name:"hla_gr_over" ~n:(n + 1) ~heap:true;
  grid_leg ~what:"12 MiB" ~name:"hla_gr_big" ~n:big_elems ~heap:true

(* gh-ocannl-1877: which of a chunk's tiles keeps the stack is the program's choice, not the node
   uids'. Two equal-sized tiles, each within the chunk's budget and together over it, so the budget
   takes one: the one the chunk touches first, declared first. The renderer collected a loop's
   locals in a table keyed by uid and privatized them in the table's order, so the unrelated nodes
   minted before the routine -- [mint] of them here -- decided the order of the declarations and
   which tile went to the heap. The tile touched second is created first, so creation order is not
   the program's either. *)
let pair_leg ~mint =
  for _ = 1 to mint do
    ignore (mk "unrelated" : Tn.t)
  done;
  let name = Printf.sprintf "hla_gr_pair%d" mint in
  let n = floats (Ir.C_syntax.chunk_stack_locals_bytes * 3 / 4) in
  let second = scratch (name ^ "_second") n in
  let first = scratch (name ^ "_first") n in
  let out = mk ~dims:[| grid_extent; n |] (name ^ "_out") in
  let lead = mk ~dims:[| grid_extent |] (name ^ "_lead") in
  materialize out;
  materialize lead;
  let g = sym () and x1 = sym () and x2 = sym () and y = sym () and z = sym () in
  let llc =
    seq
      (loop_n z grid_extent (set lead [| iter z |] (tick z)))
      (loop_n ~axis:LL.Grid g grid_extent
         (seq
            (loop_n x1 n (set first [| iter x1 |] (tag g x1)))
            (seq
               (loop_n x2 n (set second [| iter x2 |] (add (c 100.) (tag g x2))))
               (loop_n y n
                  (set out
                     [| iter g; iter y |]
                     (add (get first [| rev ~n y |]) (get second [| rev ~n y |])))))))
  in
  let want =
    Array.init (grid_extent * n) ~f:(fun cell ->
        let g = cell / n and x = cell % n in
        Float.of_int (100 + (2 * (1 + (10 * g) + (n - 1 - x)))))
  in
  let src, got, want = run_leg ~name (optimize ~materialized:[ out ] ~name llc, out, want) in
  (* The renderer's table, replayed: keyed by uid, filled in access order. *)
  let table_first_ahead =
    let table = Hashtbl.create (module Int) in
    Hashtbl.set table ~key:first.Tn.uid ~data:();
    Hashtbl.set table ~key:second.Tn.uid ~data:();
    List.hd_exn (Hashtbl.keys table) = first.Tn.uid
  in
  let line ident = List.map (declarations src ~ident) ~f:(fun a -> a.Generated.line) in
  ( table_first_ahead,
    on_stack src ~scope:`Block ~ident:(name ^ "_first") ~n
    && on_heap src ~scope:`Block ~ident:(name ^ "_second") ~n,
    (match (line (name ^ "_first"), line (name ^ "_second")) with
    | [ l1 ], [ l2 ] -> l1 < l2
    | _ -> false),
    Array.equal Float.equal got want )

let () =
  let builds = List.map [ 0; 1; 2; 3; 4; 5 ] ~f:(fun mint -> pair_leg ~mint) in
  (* Without both orders, a table-ordered choice would pass too. *)
  p
    "Grid chunk, two tiles over the budget together: control: a table keyed by their uids has the \
     first-touched tile ahead in one build and behind in another"
    (List.exists builds ~f:(fun (ahead, _, _, _) -> ahead)
    && List.exists builds ~f:(fun (ahead, _, _, _) -> not ahead));
  p_all
    "Grid chunk, two tiles over the budget together: the first-touched keeps the stack, the other \
     takes the heap, whatever was minted before"
    builds ~f:(fun (_, placed, _, _) -> placed);
  p_all "Grid chunk, two tiles over the budget together: declared in the order touched" builds
    ~f:(fun (_, _, ordered, _) -> ordered);
  p_all "Grid chunk, two tiles over the budget together: values" builds ~f:(fun (_, _, _, v) -> v)
