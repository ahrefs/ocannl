(* gh-ocannl-1675: the prelude assembly ([C_syntax.prepend_builtins_to_text]) no longer scans every
   kernel's whole source a second time to pick its conditional includes; it joins the scans of the
   parts it concatenates -- the includes, each builtin definition, the kernels -- by
   [Source_tokens.joined], and selects builtins through [Source_tokens.mentions_indexed]. Both are
   pinned here against the plain scan of the joined text, the reading they replace, over the real
   builtin tables and over the joins where a part ends inside a comment or a literal: there the
   parts' scans are not the joined text's, and [joined] has to decline. *)

open Base
open Verdict.Claims
module T = Ir.C_syntax.Source_tokens

let definitions builtins = List.map builtins ~f:(fun (_, definition, _) -> definition)
let keys builtins = List.map builtins ~f:(fun (key, _, _) -> key)

(* A kernel-like text that uses some builtins of each table, with a header marker, a comment and a
   literal naming others. *)
let kernel =
  {|kernel void k(device float *x) {
  /* uses_nothing_here: HALF_T */ x[0] = fast_exp(x[1]) + simdgroup_load(x);
  printf("HALF_TO_FLOAT %d", 1); // ocannl_cp_async4_probe
}|}

(* Markers that cross part boundaries, single and multi-token, present and absent. *)
let straddling = [ "#include <metal_simdgroup_matrix>"; "x #include"; "> y"; "rocwmma::"; "a b" ]

let corpora =
  [
    ("cc builtins", true, (Builtins_cc.includes :: definitions Builtins_cc.builtins) @ [ kernel ]);
    ("metal builtins", true, definitions Builtins_metal.builtins @ [ kernel ]);
    ("a marker straddling two parts", true, [ "x #include"; "<metal_simdgroup_matrix> y" ]);
    ("a line comment ending a part", true, [ "a // simdgroup_load"; "rocwmma"; "::" ]);
    ("a block comment open across parts", false, [ "x /* open"; "simdgroup_load */ y" ]);
    ("a literal open across parts", false, [ {|s = "unterminated|}; {|simdgroup_load" a b|} ]);
  ]

let markers =
  List.dedup_and_sort ~compare:String.compare
    (keys Builtins_cc.builtins @ keys Builtins_metal.builtins @ straddling
    @ [ "simdgroup_load"; "fast_exp"; "HALF_T"; "HALF_TO_FLOAT"; "ocannl_cp_async4_probe" ])

let () =
  List.iter corpora ~f:(fun (label, closed, parts) ->
      let whole = T.scan (String.concat ~sep:"\n" parts) in
      let indexed = T.mentions_indexed whole in
      p_all ("mentions_indexed answers mentions: " ^ label) markers ~f:(fun m ->
          Bool.equal (indexed m) (T.mentions whole m));
      match T.joined (List.map parts ~f:T.scan) with
      | Some joined ->
          p ("joined answers for the parts: " ^ label) closed;
          p_all ("joined mentions what the joined text mentions: " ^ label) markers ~f:(fun m ->
              Bool.equal (T.mentions joined m) (T.mentions whole m))
      | None -> p ("joined declines the parts: " ^ label) (not closed));
  (* The pins above are not vacuous: the real tables' kernel mentions some of their builtins and the
     straddling marker, and the open joins are ones where the parts' own scans would read a marker
     the joined text does not have. *)
  let cc_whole =
    T.scan
      (String.concat ~sep:"\n"
         ((Builtins_cc.includes :: definitions Builtins_cc.builtins) @ [ kernel ]))
  in
  p_exists "the cc corpus mentions some builtin key" (keys Builtins_cc.builtins)
    ~f:(T.mentions cc_whole);
  p "a marker straddling two parts is mentioned"
    (T.mentions
       (T.scan "x #include\n<metal_simdgroup_matrix> y")
       "#include <metal_simdgroup_matrix>");
  p "an open block comment hides what its part's own scan reads"
    (T.mentions (T.scan "simdgroup_load */ y") "simdgroup_load"
    && not (T.mentions (T.scan "x /* open\nsimdgroup_load */ y") "simdgroup_load"));
  (* [scan_static] is [scan], remembered by content: a copy of the text, as a backend that rebuilds
     its includes for every render passes, is answered by the same scan. *)
  let static = T.scan_static Builtins_cc.includes in
  let copy = Bytes.to_string (Bytes.of_string Builtins_cc.includes) in
  p "scan_static remembers a text by content"
    ((not (phys_equal copy Builtins_cc.includes)) && phys_equal static (T.scan_static copy));
  p_all "scan_static mentions what scan does" markers ~f:(fun m ->
      Bool.equal (T.mentions static m) (T.mentions (T.scan Builtins_cc.includes) m))
