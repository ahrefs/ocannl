(* staging#2147: wider generated names can wrap a kernel's pretty source before the cache
   canonicalizes those names. Render both layouts on the host without invoking a C compiler. *)
open Base
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode
module B = Ll_builders

module Driver =
  Ir.C_syntax.Compile_driver
    (functor
       (Input : sig
          val procs : LL.t array
        end)
       ->
       Ir.C_syntax.Pure_C_config (struct
         include Input

         let full_printf_support = false
       end))

let render (id : LL.scope_id) =
  (* Repetition makes the scope id's width matter to the sum without relying on global counters or
     on the width of a single declaration. *)
  let local = LL.Get_local id in
  let sum = List.fold (List.init 9 ~f:(fun _ -> local)) ~init:local ~f:B.add in
  let llc = B.seq (B.declare_local ~needs_init:true id) (LL.Set_local (id, sum)) in
  let proc : LL.optimized =
    {
      traced_store = Hashtbl.create (module Tn);
      optimize_ctx = LL.empty_optimize_ctx ();
      llc;
      merge_node = None;
      workgroup_shared = Set.empty (module Tn);
      simdgroup_fragments = Set.empty (module Tn);
      masked_fragments = Set.empty (module Tn);
      swizzled = Map.empty (module Tn);
      pipelined = Map.empty (module Tn);
      zero_fringe = Set.empty (module Tn);
      flip_candidates = [];
      spliced_rbw = Set.empty (module Tn);
      source = llc;
    }
  in
  Driver.render_batch ~names:[| "cache_key_wrapping" |] Ir.Indexing.Empty [| proc |] ~includes:""
    ~builtins:[] ()

let () =
  let tn = B.node_factory ~first_id:0 ~dims:[||] () "z" in
  B.virtualize tn;
  let short = render { tn; scope_id = 9 } in
  let wide = render { tn; scope_id = 10000 } in
  let pretty_lines batch = List.length (String.split_lines (batch.Driver.assemble [ 0 ])) in
  p "wider scope names add a line break to the pretty source"
    (pretty_lines wide > pretty_lines short);
  p "kernels across the pretty wrap boundary have equal cache keys"
    (String.equal (short.key_text 0) (wide.key_text 0))
