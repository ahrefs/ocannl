open Base
open Verdict.Claims
module Cs = Ir.C_syntax
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Idx = Ir.Indexing

module Config (Input : sig
  val procs : LL.t array
end) =
struct
  include Cs.Pure_C_config (struct
    let procs = Input.procs
    let full_printf_support = true
  end)
end

module Driver = Cs.Compile_driver (Config)

module Syntax = Cs.C_syntax (Config (struct
  let procs = [||]
end))

let optimized llc =
  {
    LL.traced_store = Hashtbl.create (module Tn);
    llc;
    optimize_ctx = LL.empty_optimize_ctx ();
    merge_node = None;
    workgroup_shared = Set.empty (module Tn);
    simdgroup_fragments = Set.empty (module Tn);
    swizzled = Map.empty (module Tn);
    pipelined = Map.empty (module Tn);
    zero_fringe = Set.empty (module Tn);
    flip_candidates = [];
    spliced_rbw = Set.empty (module Tn);
  }

let headers =
  [
    ("nvcuda::wmma", "#include <mma.h>");
    ("rocwmma::", "#include <rocwmma/rocwmma.hpp>");
    ("simdgroup_load", "#include <metal_simdgroup_matrix>");
  ]

let prepare text =
  Syntax.filter_and_prepend_builtins ~conditional_includes:headers ~routine_names:[]
    ~includes:"BASE" ~builtins:[] ~proc_doc:(PPrint.string text) ()

let () =
  p_all "vendor headers follow real identifier and namespace tokens" headers
    ~f:(fun (marker, header) ->
      String.is_prefix (prepare (marker ^ "(x);")) ~prefix:(header ^ "\nBASE\n"));
  let decoys =
    List.concat_map headers ~f:(fun (marker, _) ->
        [
          "prefix_" ^ marker;
          "/* " ^ marker ^ " */";
          "// " ^ marker ^ "\n";
          "printf(\"" ^ marker ^ "\");";
          "'" ^ marker ^ "'";
        ])
    @ [
        "nvcuda::wmma_suffix";
        "simdgroup_load_suffix";
        {|printf("escaped \" simdgroup_load");|};
        {|nvcuda "literal" :: wmma|};
      ]
  in
  p_all "longer identifiers comments and literals cannot inject vendor headers" decoys
    ~f:(fun source -> String.is_prefix (prepare source) ~prefix:"BASE\n");
  p "qualified tokens match across spaces and comments"
    (String.is_prefix
       (prepare "nvcuda /* namespace */ :: wmma::load_matrix_sync(x);")
       ~prefix:"#include <mma.h>\n");
  p "a commented include cannot suppress a required header"
    (String.is_prefix
       (prepare "/*\n#include <mma.h>\n*/ nvcuda::wmma::fragment x;")
       ~prefix:"#include <mma.h>\n");
  let prepared = prepare "rocwmma::fragment x;" in
  p "direct compiler preparation is idempotent"
    (String.equal prepared (Cs.prepend_conditional_includes ~conditional_includes:headers prepared));
  let builtins = [ ("dependency", "DEPENDENCY", []); ("helper", "HELPER", [ "dependency" ]) ] in
  let source =
    Syntax.filter_and_prepend_builtins ~routine_names:[] ~includes:"BASE" ~builtins
      ~proc_doc:(PPrint.string "helper(x);") ()
  in
  p "builtin dependencies retain declaration order"
    (String.equal source "BASE\nDEPENDENCY\nHELPER\nhelper(x);");
  p "builtin decoys share the vendor token scan"
    (String.equal
       (Syntax.filter_and_prepend_builtins ~routine_names:[] ~includes:"BASE" ~builtins
          ~proc_doc:(PPrint.string {|helper_suffix(); /* helper */ puts("helper");|})
          ())
       {|BASE
helper_suffix(); /* helper */ puts("helper");|});
  let first, bindings = Idx.get_static_symbol Idx.Empty in
  let second, bindings = Idx.get_static_symbol bindings in
  let proc = optimized (LL.Comment "first procedure") in
  let calls = ref [] in
  let owner = ref 770 in
  let compile_source ~name source =
    calls := (name, source) :: !calls;
    owner
  in
  let result, kparams, name, _ =
    Driver.compile ~name:"asm" bindings proc ~includes:"BASE" ~builtins:[] ~compile_source ()
  in
  p "single compiler receives the sanitized emitted symbol once"
    (String.equal name "asm__"
    && List.length !calls = 1
    && String.equal (fst (List.hd_exn !calls)) name);
  p "single compile returns the original compiler-owned object" (phys_equal result owner);
  let actual_symbols =
    List.filter_map kparams ~f:(function _, Ir.Backend_intf.Static_idx s -> Some s | _ -> None)
  in
  p_all2 "static parameters retain binding order" (Array.of_list actual_symbols) [| first; second |]
    ~f:Idx.equal_static_symbol;
  calls := [];
  let result, entries =
    Driver.compile_batch
      ~names:[| "batch_first"; "batch_second" |]
      bindings
      [| proc; optimized (LL.Comment "second procedure") |]
      ~includes:"BASE" ~builtins:[] ~compile_source ()
  in
  p "batch compiles one source and shares compiler ownership"
    (List.length !calls = 1 && phys_equal result owner);
  let batch_name, source = List.hd_exn !calls in
  p "batch artifact uses the trimmed common prefix" (String.equal batch_name "batch");
  p_all2 "batch metadata preserves routine order"
    (Array.map entries ~f:(fun (_, name, _) -> name))
    [| "batch_first"; "batch_second" |]
    ~f:String.equal;
  p "batch source preserves procedure order"
    (Option.value_exn (String.substr_index source ~pattern:"first procedure")
    < Option.value_exn (String.substr_index source ~pattern:"second procedure"));
  let failure = Failure "injected compiler failure" in
  let caught =
    try
      ignore
        (Driver.compile_batch ~names:[| "a"; "b" |] bindings [| proc; proc |] ~includes:"BASE"
           ~builtins:[]
           ~compile_source:(fun ~name:_ _ -> raise failure)
           ());
      None
    with exn -> Some exn
  in
  p "compiler failure crosses the driver without replacing its identity"
    (Option.exists caught ~f:(phys_equal failure));
  let options = [ "--fast-math"; "--no-reassociate" ] in
  let enriched =
    try
      Cs.with_compiler_options ~compiler:"vendor" ~options
        ~enrich:(fun exn ~suffix ->
          match exn with Failure msg -> Some (Failure (msg ^ suffix)) | _ -> None)
        (fun () -> raise failure)
    with Failure msg -> msg
  in
  p "failure context appends the exact effective option vector"
    (String.equal enriched
       ("injected compiler failure\nvendor options: " ^ Ir.Compiler_options.render options))
