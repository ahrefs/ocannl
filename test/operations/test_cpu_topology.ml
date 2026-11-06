(* Tests for the CPU topology facts and the worker-pool uniformity decision (gh-ocannl-530,
   docs/design/gh-ocannl-530-pool-uniformity.md). The parsers and the decision function are pure, so
   they are exercised on fabricated inputs (rog's recorded topology among them) with
   machine-independent golden output; the live probes are only checked for invariants, with the
   machine's actual facts printed to stderr for manual inspection. *)

open Base
open Stdio
module CT = Utils.Cpu_topology

let sexp_str s = Sexp.to_string_hum s

(* rog's recorded native topology (benchmarks/report-gh530-rog-pinning.md): P-cores are LPs
   {0,1,10-13,22,23} = mask 0xC03C03, E-cores the other sixteen = 0x3FC3FC. *)
let rog_p_list = "0-1,10-13,22-23"
let rog_e_list = "2-9,14-21"
let rog_classes = CT.parse_classes_str "1:8:c03c03;0:16:3fc3fc"
let mac_style_classes = CT.parse_classes_str "1:12:0;0:4:0"
let uniform_classes = CT.parse_classes_str "0:16:ffff"

let () =
  printf "== parse_cpu_list ==\n";
  List.iter [ rog_p_list; rog_e_list; "0-15"; "3"; ""; "0-,2"; "5-3"; "1,foo" ] ~f:(fun s ->
      printf "%S -> %s\n" s (sexp_str ([%sexp_of: int list option] (CT.parse_cpu_list s))));
  printf "\n== mask_of_cpu_list ==\n";
  let mask_of s =
    Option.bind (CT.parse_cpu_list s) ~f:CT.mask_of_cpu_list
    |> Option.value_map ~default:"None" ~f:(Printf.sprintf "0x%Lx")
  in
  printf "%S -> %s\n" rog_p_list (mask_of rog_p_list);
  printf "%S -> %s\n" rog_e_list (mask_of rog_e_list);
  printf "%S -> %s\n" "64" (mask_of "64");
  printf "[] -> %s\n"
    (CT.mask_of_cpu_list [] |> Option.value_map ~default:"None" ~f:(Printf.sprintf "0x%Lx"));
  printf "\n== parse_classes_str ==\n";
  List.iter
    [
      "1:8:c03c03;0:16:3fc3fc";
      "0:16:3fc3fc;1:8:c03c03" (* order-independent: sorted fastest-first *);
      "1:12:0;0:4:0" (* Darwin style: counts only *);
      "1:8" (* malformed *);
      "1:0:ff" (* zero count *);
      "";
    ] ~f:(fun s ->
      printf "%S -> %s\n" s (sexp_str ([%sexp_of: CT.core_class list] (CT.parse_classes_str s))));
  printf "\n== classes_of_ranked_cpu_lists ==\n";
  let ranked =
    [
      (1, Option.value_exn (CT.parse_cpu_list rog_p_list));
      (0, Option.value_exn (CT.parse_cpu_list rog_e_list));
    ]
  in
  printf "rog -> %s\n"
    (sexp_str ([%sexp_of: CT.core_class list] (CT.classes_of_ranked_cpu_lists ranked)));
  printf "with cpu 64 -> %s\n"
    (sexp_str
       ([%sexp_of: CT.core_class list] (CT.classes_of_ranked_cpu_lists [ (1, [ 0 ]); (0, [ 64 ]) ])));
  printf "\n== decide_pool_restriction ==\n";
  let show ?affinity_mask ~openmp ~setting ~classes ~hypervisor ~effective label =
    let d =
      CT.decide_pool_restriction ~openmp ~setting ~classes ~hypervisor ~effective ~affinity_mask
    in
    printf "%-42s -> restrict=%s width=%d tag=%s\n" label
      (Option.value_map d.CT.pool_restrict ~default:"no" ~f:(fun c ->
           Printf.sprintf "0x%Lx" c.CT.mask))
      d.CT.pool_width d.CT.pool_tag
  in
  (* Meteor-Lake-style three classes: P / E / low-power island. `Efficiency must pick the middle
     class, not the near-serial LP-E island. *)
  let three_classes = CT.parse_classes_str "2:8:ff;1:8:ff00;0:2:30000" in
  show ~openmp:true ~setting:`Auto ~classes:rog_classes ~hypervisor:`No ~effective:24
    "rog native, auto";
  show ~openmp:true ~setting:`Performance ~classes:rog_classes ~hypervisor:`No ~effective:24
    "rog native, performance";
  show ~openmp:true ~setting:`Efficiency ~classes:rog_classes ~hypervisor:`No ~effective:24
    "rog native, efficiency";
  show ~openmp:true ~setting:`Efficiency ~classes:three_classes ~hypervisor:`No ~effective:18
    "three classes, efficiency";
  show ~openmp:true ~setting:`All ~classes:rog_classes ~hypervisor:`No ~effective:24
    "rog native, all";
  show ~openmp:true ~setting:`Auto ~classes:rog_classes ~hypervisor:`Yes ~effective:24
    "rog under WSL2 (fabricated topology), auto";
  show ~openmp:true ~setting:`Auto ~classes:rog_classes ~hypervisor:`Unknown ~effective:24
    "hypervisor unknown, auto";
  show ~openmp:true ~setting:`Performance ~classes:rog_classes ~hypervisor:`Unknown ~effective:24
    "hypervisor unknown, explicit performance";
  show ~openmp:false ~setting:`Auto ~classes:rog_classes ~hypervisor:`No ~effective:24
    "libdispatch pool, auto";
  show ~openmp:true ~setting:`Auto ~classes:uniform_classes ~hypervisor:`No ~effective:16
    "uniform machine, auto";
  show ~openmp:true ~setting:`Auto ~classes:mac_style_classes ~hypervisor:`No ~effective:16
    "classes without masks, auto";
  show ~openmp:true ~setting:`Auto ~classes:[] ~hypervisor:`No ~effective:8 "no class info, auto";
  show ~openmp:true ~setting:`Auto ~classes:rog_classes ~hypervisor:`No ~effective:8
    ~affinity_mask:0xC03C03L "externally pinned to 8P, auto";
  show ~openmp:true ~setting:`Auto ~classes:rog_classes ~hypervisor:`No ~effective:8
    ~affinity_mask:0x3FCL "externally pinned to 8E, auto";
  show ~openmp:true ~setting:`Performance ~classes:rog_classes ~hypervisor:`No ~effective:8
    ~affinity_mask:0x3FCL "externally pinned, performance";
  printf "\n== live probe invariants ==\n";
  let classes = CT.core_classes () in
  let effective = CT.effective_cpu_count () in
  Verdict.p "effective_cpu_count >= 1" (effective >= 1);
  Verdict.p "classes well-formed"
    (List.is_sorted_strictly classes ~compare:(fun a b -> Int.compare b.CT.perf_rank a.CT.perf_rank)
    && List.for_all classes ~f:(fun c -> c.CT.count > 0)
    && List.for_all classes ~f:(fun c ->
        Int64.(c.CT.mask = 0L) || Int64.popcount c.CT.mask = c.CT.count)
    &&
    (* Nonzero masks are pairwise disjoint. *)
    let masked = List.filter classes ~f:(fun c -> Int64.(c.CT.mask <> 0L)) in
    List.for_alli masked ~f:(fun i a ->
        List.for_alli masked ~f:(fun j b -> i = j || Int64.(a.CT.mask land b.CT.mask = 0L))));
  (* The machine's actual facts, for manual inspection only (stderr is not golden-checked). *)
  eprintf "machine classes: %s\n" (sexp_str ([%sexp_of: CT.core_class list] classes));
  eprintf "machine effective_cpu_count: %d\n" effective;
  eprintf "machine hypervisor: %s\n"
    (match CT.hypervisor_present () with `Yes -> "yes" | `No -> "no" | `Unknown -> "unknown");
  (* Hardware-effect hook, off in golden runs: with OCANNL_TOOL_TEST_RESTRICT_MASK=<hex> set, apply
     the process restriction and report (stderr) the affinity-respecting count before/after —
     verifying the setter stub end-to-end on a real machine, e.g. w8P on the gh-530 rog box via
     OCANNL_TOOL_TEST_RESTRICT_MASK=c03c03. *)
  match Stdlib.Sys.getenv_opt "OCANNL_TOOL_TEST_RESTRICT_MASK" with
  | None | Some "" -> ()
  | Some hex -> (
      let mask = Int64.of_string ("0x" ^ hex) in
      eprintf "restrict to 0x%Lx: before=%d " mask effective;
      match CT.restrict_process_to_mask mask with
      | Ok () -> eprintf "-> after=%d\n" (CT.effective_cpu_count ())
      | Error msg -> eprintf "-> error: %s\n" msg)
