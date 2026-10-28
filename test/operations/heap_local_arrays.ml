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
   x]. *)
let grid_routine ~name tile ~n =
  let out = mk ~dims:[| grid_extent; n |] (name ^ "_out") in
  materialize out;
  let g = sym () and x = sym () and y = sym () in
  let llc =
    loop_n ~axis:LL.Grid g grid_extent
      (seq
         (loop_n x n (set tile [| iter x |] (tag g x)))
         (loop_n y n (set out [| iter g; iter y |] (get tile [| rev ~n y |]))))
  in
  let want =
    Array.init (grid_extent * n) ~f:(fun cell ->
        let g = cell / n and x = cell % n in
        Float.of_int (1 + (10 * g) + (n - 1 - x)))
  in
  (optimize ~materialized:[ out ] ~name llc, out, want)

(* The emitted forms of a tile, by its identifier (its label: each is unique in its routine and
   clear of every reserved name; every tile is single precision): a stack array declares [float
   ident[n]], a heap array a [float *] named [ident] into the allocation [free] takes, named by the
   renderer's own derivation. Each form is claimed present, never only the other absent, so an
   identifier spelled otherwise fails rather than passes. *)
let allocation ident = Ir.C_syntax.derived_name Ir.C_syntax.heap_local_derivation ident
let declared_on_stack src ~ident = String.is_substring src ~substring:("float " ^ ident ^ "[")

let on_stack src ~ident ~n =
  String.is_substring src ~substring:(Printf.sprintf "float %s[%d]" ident n)
  && not (String.is_substring src ~substring:(allocation ident))

let on_heap src ~ident =
  String.is_substring src ~substring:(Printf.sprintf "float * const %s = " ident)
  && String.is_substring src ~substring:(Printf.sprintf "free(%s);" (allocation ident))
  && not (declared_on_stack src ~ident)

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
        pf "function scope, %s: %s is a heap array, freed" what label (on_heap src ~ident:label)
      else
        pf "function scope, %s: %s is a stack array" what label
          (on_stack src ~ident:label ~n:(Tn.num_elems tn)));
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
  p "function scope, beside a node labeled free: the heap array, freed" (on_heap src ~ident:name);
  p "function scope, beside a node labeled free: no array is declared as free"
    (not (declared_on_stack src ~ident:"free"));
  p_all2 "function scope, beside a node labeled free: values" got want ~f:Float.equal

let grid_leg ~what ~name ~n ~heap =
  let tile = scratch (name ^ "_tile") n in
  let src, got, want = run_leg ~name (grid_routine ~name tile ~n) in
  let id = name ^ "_tile" in
  pf "Grid chunk, %s: the Grid renders on the pool" what (pool_rendered src);
  if heap then pf "Grid chunk, %s: a heap tile, freed" what (on_heap src ~ident:id)
  else pf "Grid chunk, %s: a stack tile" what (on_stack src ~ident:id ~n);
  pf_all2 "Grid chunk, %s: values" what got want ~f:Float.equal

let () =
  let n = floats Ir.C_syntax.chunk_stack_locals_bytes in
  grid_leg ~what:"at the budget" ~name:"hla_gr_at" ~n ~heap:false;
  grid_leg ~what:"one element over the budget" ~name:"hla_gr_over" ~n:(n + 1) ~heap:true;
  grid_leg ~what:"12 MiB" ~name:"hla_gr_big" ~n:big_elems ~heap:true
