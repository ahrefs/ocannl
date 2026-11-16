(* Host-compiled probes of integer RNG helper results, before storage conversion. Reads only
   uint4x32_to_{byte,uint16,int32,uint32,int64,uint64}_uniform{,_vec,_lane} names. Generic C
   arithmetic promotions and parameter guards are outside this contract. *)
let formats =
  [
    ("byte", 8, false);
    ("uint16", 16, false);
    ("int32", 32, true);
    ("uint32", 32, false);
    ("int64", 64, true);
    ("uint64", 64, false);
  ]

let expected_keys =
  List.concat_map
    (fun (prec, _, _) ->
      let stem = "uint4x32_to_" ^ prec ^ "_uniform" in
      List.map (fun suffix -> stem ^ suffix) [ ""; "_vec"; "_lane" ])
    formats

let has_all_forms keys = List.sort_uniq compare keys = List.sort compare expected_keys

let format key =
  List.find_opt
    (fun (prec, _, _) ->
      let stem = "uint4x32_to_" ^ prec ^ "_uniform" in
      List.mem key [ stem; stem ^ "_vec"; stem ^ "_lane" ])
    formats

let is_integer key = Option.is_some (format key)

(* [r] retains the call's actual type. Check signedness as well as widened values: a cast to
   uint64_t alone would hide a signed 64-bit return. Vector elements retain their actual type too.
   The oracle extracts input bits directly, independently of every builtin table. *)
let check key =
  match format key with
  | None -> None
  | Some (prec, bits, signed) ->
      let vec = String.ends_with ~suffix:"_vec" key in
      let lane = String.ends_with ~suffix:"_lane" key in
      (* The existing byte pack ABI is int8x16_t, unlike unsigned scalar/lane results. *)
      let signed = signed || (prec = "byte" && vec) in
      let value = if vec then "r.v[j]" else "r" in
      let index = if vec then "j" else if lane then "lane" else "0" in
      let typ = (if signed then "int" else "uint") ^ string_of_int bits ^ "_t" in
      let wide = if signed then "int64_t" else "uint64_t" in
      let source =
        if bits = 64 then
          Printf.sprintf "((uint64_t)in[2*(%s)] | ((uint64_t)in[2*(%s)+1] << 32))" index index
        else
          Printf.sprintf "(in[(%s)/%d] >> (((%s)%%%d)*%d))" index (32 / bits) index (32 / bits) bits
      in
      Some
        (Printf.sprintf
           "  for (int j = 0; j < %d; ++j) {\n\
           \    if (((__typeof__(%s))-1 < (__typeof__(%s))0) != %d) return 0;\n\
           \    if (sizeof(%s) != %d || (%s)(%s) != (%s)(%s)(%s)) return 0;\n\
           \  }\n\
           \  return 1;\n"
           (if vec then 128 / bits else 1)
           value value
           (if signed then 1 else 0)
           value (bits / 8) wide value wide typ source)

(* Mutate only one scalar return declaration, leaving the byte store unchanged. Fail closed if the
   declaration changes, so the negative control cannot silently become an unmodified copy. *)
let signed_byte_table table =
  List.map
    (fun (key, definition, deps) ->
      if key <> "uint4x32_to_byte_uniform" then (key, definition, deps)
      else
        let marker = "uint8_t " ^ key in
        let alternate = "unsigned char " ^ key in
        let replace marker =
          let n = String.length marker in
          let hits =
            List.filter
              (fun i -> String.sub definition i n = marker)
              (List.init (max 0 (String.length definition - n + 1)) Fun.id)
          in
          match hits with
          | [ i ] ->
              Some
                (String.sub definition 0 i ^ "signed char " ^ key
                ^ String.sub definition (i + n) (String.length definition - i - n))
          | _ -> None
        in
        let changed =
          match replace marker with
          | Some s -> s
          | None -> (
              match replace alternate with
              | Some s -> s
              | None -> failwith "byte return mutation did not match once")
        in
        (key, changed, deps))
    table
