(* gh-ocannl-1696: every test source that claims bitwise equality of executed values either pins
   [cc_backend_fp_contract=off] or says, in place, why the C compiler's contraction cannot move a
   bit.

   Under the default [cc_backend_fp_contract=auto], whether gcc fuses a product into the add after
   it depends on the kernel's SHAPE -- a vectorized body against its scalar remainder, a packed tile
   against the serial loop, unrolled against looped -- and on the compiler and the host: gcc before
   14 treats [-ffp-contract=on] as [off], and only the AVX512-FP16 CI runners compute f16 natively
   (gh-ocannl-1679, gh-ocannl-1682). So a bitwise claim between two kernel shapes can pass on the
   fleet's gcc 15 and fail on CI's gcc 13, or the other way round. [off] is the one mode every
   compiler reads the same way.

   {1 What it reads -- a LINT, and its boundary}

   - THE SUBJECT: a test source (test/ and arrayjit/test/, as [Ll_test_scan.test_source] has it) one
   of whose string literals -- a Verdict claim's name, in practice -- holds [bitwise], [bit-for-bit]
   or [bit-identical], in any case. Read with the compiler's lexer ({!Prose_lexemes}), so a comment
   saying "bitwise" is not a claim. Also reads the closed code forms below through ppxlib and
   Lexical_scope.

   Recognised forms (closed): Int64.equal or Int32.equal with at least one operand summary reaching
   Int64.bits_of_float or Int32.bits_of_float, including comparison to a literal bit encoding;
   Array.equal/List.equal with such a predicate; local functions with known arity and aliases.
   Numeric Float.equal alone is outside this bit-pattern extension.

   Claims are derived from Verdict.Claims exports and the existing Verdict_scan.leg_population_slots
   model, whose coverage verdict_ratchet checks against those exports. Boolean_at/Boolean_last read
   the specified positional argument; Collections_at/Collections_last read those collections,
   together with labelled f/equal predicates and over populations. Thus formatted forms read their
   trailing value(s), not formatting arguments. Qualified Verdict.<name>/Verdict.Claims.<name>,
   module aliases, and open/include Verdict or Verdict.Claims use lexical resolution. Slotless
   exports (message, skip, case boundary and leg wrappers) make no population claim themselves;
   claims nested within their callbacks are still visited.

   Output origins: Context.get_values with its context and tensor, and Ll_test.execute or
   Ll_test.execute_with_census with an optimized program and labelled read. A recognised claim must
   reach both an applied comparator and an output-read dependency. Summaries union expression
   children and tuple bindings: conservative dependency accounting, not per-element provenance.
   Ordinary function parameters ([fun x ->] or [let f x =]) union the arguments of fully applied
   lexical calls in this file, to a finite monotone fixpoint (recursive calls included).
   Unconstrained nested ordinary lambdas are flattened to count their full arity; returning a
   comparator identifier does not execute it. Escaping helpers receive no evidence from unknown
   external callers; uncalled parameters stay unknown. The List/Array iter, iteri, map, mapi,
   for_all, for_alli, iter2[_exn] and map2[_exn] table maps collection dependencies into inline
   labelled f callbacks with ordinary parameters, requiring every mapped positional collection
   before invoking a callback, and treating a fully saturated comparator callback as applied (index
   parameters stay unknown), including tuple patterns. Tuple components share their collection's
   union; this is not per-element provenance. Other higher-order calls, mutable storage, external
   helper bodies and exported module-structure values remain outside this extension. Calls have an
   identifier claim callee and operands in one application node; operator-based application, partial
   helper applications, function-case parameters, omitted optional defaults and bare external-reader
   aliases are not reconstructed.

   - A PIN: the string literal [OCANNL_CC_BACKEND_FP_CONTRACT] whose next string literal is [off]
   (the [Unix.putenv] the pinned tests do at module initialization). A pin made elsewhere -- a dune
   stanza's flag or profile, a directory's ocannl_config -- is not read: such a source carries a
   marker instead, or pins in-process too. - A MARKER: a comment whose whole text is
   [cc-fp-contract], a colon, a kind, [--] and a reason of at least two words. Every comment holding
   the sentinel followed by a colon must be one, so a misspelt kind or a missing reason fails rather
   than reads as absent. The kinds, a closed vocabulary ([kinds] below), each name why contraction
   cannot reach the claim.

   Every subject is listed in the golden with its disposition, so the list is the review surface: a
   new bitwise claim lands with its pin or its marker in the same diff. A source with a pin AND a
   marker, two markers, or a marker but no claim fails too, so a disposition never outlives the
   claim it was written for. *)
open Base
open Stdio
open Verdict.Claims
module Dune = Test_scan.Dune_stanza_scan
module Sources = Test_scan.Config_key_scan
open Ppxlib
module Scope = Test_scan.Lexical_scope

(* cc-fp-contract: not-executed -- its bitwise text is fixture sources and claim names, no
   kernels *)

let kinds =
  [
    ( "exact-operands",
      "every product and partial sum is exact at the compared precision, so a fused multiply-add \
       rounds where the separate ones do: nowhere" );
    ( "no-contraction",
      "the compared values involve no floating product feeding an add: nothing for the compiler to \
       fuse" );
    ("same-shape", "on cc, both sides run the same kernel code, so they contract alike");
    ( "gated",
      "the claim reads the kernel's flags and target at run time and skips where they differ" );
    ("not-cc", "only a GPU backend evaluates the claim; cc skips it");
    ("not-executed", "no claim is about executed values: fixtures, codegen text, host arithmetic");
  ]

let sentinel = "cc-fp-contract:"

(* The pin's two literals, the value spelled first: the other order would make this file's own
   source read as a pin. *)
let pin_value = "off"
let pin_key = "OCANNL_CC_BACKEND_FP_CONTRACT"
let subject_phrases = [ "bitwise"; "bit-for-bit"; "bit-identical" ]

type disposition = Pinned | Marked of string * string

type reading = {
  claim_lines : int list;
  pinned : bool;
  markers : (int * (string * string, string) Result.t) list;
}

(* [text] is a marker's whole comment text; [Error] says why it is not a well-formed one. *)
let parse_marker text =
  let text = String.strip text in
  match String.chop_prefix text ~prefix:sentinel with
  | None -> Error "the sentinel must open the comment"
  | Some rest -> (
      match String.substr_index rest ~pattern:" -- " with
      | None -> Error "expected `<kind> -- <reason>` after the sentinel"
      | Some i ->
          let kind = String.strip (String.prefix rest i) in
          (* A reason may wrap across lines; it is read, and listed, as its words. *)
          let reason =
            String.drop_prefix rest (i + 4)
            |> String.split_on_chars ~on:[ ' '; '\n'; '\t'; '\r' ]
            |> List.filter ~f:(Fn.non String.is_empty)
            |> String.concat ~sep:" "
          in
          if not (List.Assoc.mem kinds ~equal:String.equal kind) then
            Error ("unknown kind `" ^ kind ^ "`")
          else if List.length (String.split reason ~on:' ') < 2 then
            Error "the reason needs at least two words"
          else Ok (kind, reason))

(* Dependency summaries are lexical, not a file-wide search for a familiar helper name. *)
type evidence = {
  comparator : bool;
  compared : bool;
  output : bool;
  bits : bool;
  claim : bool;
  helpers : int list;
  call_compared : bool;
}

let unknown =
  {
    comparator = false;
    compared = false;
    output = false;
    bits = false;
    claim = false;
    helpers = [];
    call_compared = false;
  }

let union a b =
  {
    comparator = a.comparator || b.comparator;
    compared = a.compared || b.compared;
    output = a.output || b.output;
    bits = a.bits || b.bits;
    claim = false;
    call_compared = a.call_compared || b.call_compared;
    helpers = List.dedup_and_sort (a.helpers @ b.helpers) ~compare:Int.compare;
  }

(* The export set is read from Verdict itself; argument positions come from the existing population
   model, whose coverage verdict_ratchet holds equal to those exports. *)
module Verdict_scan = Test_scan.Verdict_scan

let claim_slots = ref (Map.empty (module String))

let load_claim_slots content =
  let exports =
    Test_scan.Refusal_control_scan.claims_members content
    |> Option.value_exn ~message:"Verdict.Claims exports could not be read"
  in
  List.fold exports
    ~init:(Map.empty (module String))
    ~f:(fun slots name ->
      match List.Assoc.find Verdict_scan.leg_population_slots name ~equal:String.equal with
      | Some slot -> Map.set slots ~key:name ~data:slot
      | None when List.mem Verdict_scan.leg_slotless_members name ~equal:String.equal -> slots
      | None -> failwith ("Verdict.Claims member has no argument model: " ^ name))

(* One mapping per family, shared by both modules. Integers select positional collections; None is
   an iterator index rather than an element. Unknown paths receive no mapping. *)
let callback_slots path =
  List.find_map [ "List."; "Array." ] ~f:(fun prefix ->
      Option.bind (String.chop_prefix path ~prefix) ~f:(fun name ->
          List.Assoc.find
            [
              ("iter", [ Some 0 ]);
              ("map", [ Some 0 ]);
              ("iteri", [ None; Some 0 ]);
              ("mapi", [ None; Some 0 ]);
              ("for_all", [ Some 0 ]);
              ("for_alli", [ None; Some 0 ]);
              ("iter2", [ Some 0; Some 1 ]);
              ("iter2_exn", [ Some 0; Some 1 ]);
              ("map2", [ Some 0; Some 1 ]);
              ("map2_exn", [ Some 0; Some 1 ]);
            ]
            name ~equal:String.equal))

let code_claim_lines content =
  (* A single application may supply parameters of nested ordinary lambdas too. Flatten only
     unconstrained function bodies; returned identifiers remain comparator factories. *)
  let flatten =
    object
      inherit Ast_traverse.map as super

      method! expression e =
        let e = super#expression e in
        match e.pexp_desc with
        | Pexp_function
            (params, None, Pfunction_body { pexp_desc = Pexp_function (inner, None, body); _ }) ->
            { e with pexp_desc = Pexp_function (params @ inner, None, body) }
        | _ -> e
    end
  in
  let ast = flatten#structure (Parse.implementation (Lexing.from_string content)) in
  let functions = Hashtbl.create (module Int) in
  let parameters = Hashtbl.create (module Int) in
  let changed = ref false in
  let lines = ref [] in
  let grow pattern summary =
    let key = pattern.ppat_loc.loc_start.pos_cnum in
    let old = Option.value (Hashtbl.find parameters key) ~default:unknown in
    let next = union old summary in
    if not (Poly.equal old next) then (
      changed := true;
      Hashtbl.set parameters ~key ~data:next)
  in
  let collector =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match e.pexp_desc with
        | Pexp_function (params, _, _) ->
            Hashtbl.set functions ~key:e.pexp_loc.loc_start.pos_cnum
              ~data:
                (List.filter_map params ~f:(fun p ->
                     match p.pparam_desc with
                     | Pparam_val (label, _, pat) -> Some (label, pat)
                     | Pparam_newtype _ -> None))
        | _ -> ());
        super#expression e
    end
  in
  collector#structure ast;
  let walker () =
    object (self)
      inherit [evidence, string] Scope.scoped as super
      val mutable evidence = unknown

      (* Buckets use a location only for hashing. Entries compare the environment and AST node
         physically: no polymorphic equality over Maps containing comparator closures. *)
      val cache = Hashtbl.create (module Int)

      method cached env e =
        Hashtbl.find cache e.pexp_loc.loc_start.pos_cnum
        |> Option.bind
             ~f:
               (List.find_map ~f:(fun (scope, node, summary) ->
                    if phys_equal scope env && phys_equal node e then Some summary else None))

      method! bind_parameter env _ pattern =
        let summary =
          Option.value
            (Hashtbl.find parameters pattern.ppat_loc.loc_start.pos_cnum)
            ~default:unknown
        in
        self#forget
          (Scope.bind_values env (Scope.pattern_vars [ pattern ]) summary)
          (Scope.pattern_unpacks [ pattern ])

      method local = unknown
      method! shadowed = Some "?"

      method module_path env path =
        let rec resolve = function
          | Longident.Lident name ->
              Some (Option.value (Map.find env.Scope.modules name) ~default:name)
          | Ldot (prefix, name) -> Option.map (resolve prefix) ~f:(fun p -> p ^ "." ^ name)
          | Lapply _ -> None
        in
        resolve path

      method path env e =
        match e.pexp_desc with
        | Pexp_ident { txt = Lident name; _ } ->
            if Option.is_some (Scope.lookup env name) then "?" else name
        | Pexp_ident { txt = Ldot (prefix, name); _ } ->
            Option.value_map (self#module_path env prefix) ~default:"?" ~f:(fun p -> p ^ "." ^ name)
        | _ -> "?"

      method summary env e =
        let saved = evidence in
        evidence <- unknown;
        ignore (self#expression env e : expression);
        let result = evidence in
        evidence <- saved;
        result

      method! binding_denotes env binding = self#summary env binding.pvb_expr

      method! ident env ident =
        match ident.txt with
        | Lident name ->
            Option.iter (Scope.lookup env name) ~f:(fun s -> evidence <- union evidence s)
        | _ -> ()

      method! expression env e =
        match self#cached env e with
        | Some summary ->
            evidence <- union evidence summary;
            e
        | None ->
            let saved = evidence in
            evidence <- unknown;
            let result = self#walk env e in
            let summary = evidence in
            Hashtbl.add_multi cache ~key:e.pexp_loc.loc_start.pos_cnum ~data:(env, e, summary);
            evidence <- union saved summary;
            result

      method walk env e =
        (match e.pexp_desc with
        | Pexp_apply (fn, args) ->
            let path = self#path env fn in
            let values = List.map args ~f:(fun (_, arg) -> self#summary env arg) in
            let positional =
              List.filter_map args ~f:(function Nolabel, arg -> Some arg | _ -> None)
            in
            let callback_mapping =
              Option.filter (callback_slots path) ~f:(fun slots ->
                  List.for_all slots ~f:(function
                    | None -> true
                    | Some slot -> Option.is_some (List.nth positional slot)))
            in
            let fn_summary = self#summary env fn in
            List.iter fn_summary.helpers ~f:(fun key ->
                Option.iter (Hashtbl.find functions key) ~f:(fun params ->
                    if List.length args >= List.length params then
                      let remaining = ref args in
                      List.iter params ~f:(fun (label, pattern) ->
                          let matching, rest =
                            List.partition_tf !remaining ~f:(fun (actual, _) ->
                                match (label, actual) with
                                | (Labelled a | Optional a), (Labelled b | Optional b) ->
                                    String.equal a b
                                | Nolabel, Nolabel -> true
                                | _ -> false)
                          in
                          match matching with
                          | (_, arg) :: later ->
                              remaining := later @ rest;
                              grow pattern (self#summary env arg)
                          | [] -> ())));
            Option.iter callback_mapping ~f:(fun slots ->
                List.iter args ~f:(function
                  | Labelled "f", callback ->
                      Option.iter (Hashtbl.find functions callback.pexp_loc.loc_start.pos_cnum)
                        ~f:(fun params ->
                          if List.length params <= List.length slots then
                            List.iteri params ~f:(fun i (_, pattern) ->
                                Option.iter
                                  (List.nth slots i |> Option.join)
                                  ~f:(fun slot ->
                                    Option.iter (List.nth positional slot) ~f:(fun arg ->
                                        grow pattern (self#summary env arg)))))
                  | _ -> ()));

            let is_bits =
              List.mem [ "Int64.bits_of_float"; "Int32.bits_of_float" ] path ~equal:String.equal
              && List.length positional = 1
            in
            let bit_equality =
              List.mem [ "Int64.equal"; "Int32.equal" ] path ~equal:String.equal
              && List.length values = 2
              && List.exists values ~f:(fun v -> v.bits)
            in
            let collection_equality =
              List.mem [ "Array.equal"; "List.equal" ] path ~equal:String.equal
              && match values with v :: _ -> v.comparator | [] -> false
            in
            let callback_compared =
              Option.is_some callback_mapping
              && List.exists args ~f:(function
                | Labelled "f", callback ->
                    let summary = self#summary env callback in
                    if List.is_empty summary.helpers then summary.comparator
                    else
                      summary.call_compared
                      && Option.value_map callback_mapping ~default:false ~f:(fun slots ->
                          List.exists summary.helpers ~f:(fun key ->
                              Option.value_map (Hashtbl.find functions key) ~default:false
                                ~f:(fun params -> List.length params <= List.length slots)))
                | _ -> false)
            in
            let applied =
              fn_summary.comparator
              &&
              if List.is_empty fn_summary.helpers then List.length positional >= 2
              else
                fn_summary.call_compared
                && List.exists fn_summary.helpers ~f:(fun key ->
                    Option.value_map (Hashtbl.find functions key) ~default:false ~f:(fun params ->
                        List.length args >= List.length params))
            in
            evidence <-
              union evidence
                {
                  unknown with
                  bits = is_bits;
                  output =
                    (String.equal path "Context.get_values" && List.length positional >= 2)
                    || List.mem
                         [ "Ll_test.execute"; "Ll_test.execute_with_census" ]
                         path ~equal:String.equal
                       && (not (List.is_empty positional))
                       && List.exists args ~f:(function Labelled "read", _ -> true | _ -> false);
                  comparator =
                    (collection_equality && List.length positional < 3)
                    || (fn_summary.comparator && not applied);
                  compared =
                    callback_compared || bit_equality
                    || (collection_equality && List.length positional >= 3)
                    || applied;
                };
            let slot =
              match fn.pexp_desc with
              | Pexp_ident { txt = Lident name; _ } ->
                  Option.bind (Scope.lookup env name) ~f:(fun s ->
                      if s.claim then Map.find !claim_slots name else None)
              | _ ->
                  let name =
                    match String.chop_prefix path ~prefix:"Verdict.Claims." with
                    | Some name -> Some name
                    | None -> String.chop_prefix path ~prefix:"Verdict."
                  in
                  Option.bind name ~f:(Map.find !claim_slots)
            in
            Option.iter slot ~f:(fun slot ->
                let selected =
                  match slot with
                  | Verdict_scan.Boolean_at i -> List.filter_opt [ List.nth positional i ]
                  | Boolean_last -> List.filter_opt [ List.last positional ]
                  | Collections_at indices -> List.filter_map indices ~f:(List.nth positional)
                  | Collections_last n ->
                      List.drop positional (Int.max 0 (List.length positional - n))
                in
                let summary =
                  List.fold selected ~init:unknown ~f:(fun acc arg ->
                      union acc (self#summary env arg))
                in
                let summary =
                  List.fold args ~init:summary ~f:(fun acc -> function
                    | Labelled ("f" | "equal"), arg ->
                        let s = self#summary env arg in
                        union acc { s with compared = s.compared || s.comparator }
                    | Labelled "over", arg -> union acc (self#summary env arg)
                    | _ -> acc)
                in
                if summary.compared && summary.output then
                  lines := e.pexp_loc.loc_start.pos_lnum :: !lines)
        | _ -> ());
        match e.pexp_desc with
        | Pexp_function _ ->
            let saved = evidence in
            evidence <- unknown;
            let result = super#expression env e in
            let body = evidence in
            evidence <-
              union saved
                {
                  body with
                  comparator = body.comparator || body.compared;
                  compared = false;
                  call_compared = body.compared;
                  helpers = [ e.pexp_loc.loc_start.pos_cnum ];
                };
            result
        | Pexp_apply _ ->
            let result = super#expression env e in
            evidence <- { evidence with helpers = [] };
            result
        | _ -> super#expression env e

      method! opened ~top:_ ~include_:_ env denotes =
        match denotes with
        | Some ("Verdict.Claims" | "Verdict") ->
            let names = Map.keys !claim_slots in
            Scope.push_frame env
              (Map.of_alist_exn
                 (module String)
                 (List.map names ~f:(fun name -> (name, { unknown with claim = true }))))
        | _ -> env
    end
  in
  let rec pass () =
    changed := false;
    ignore
      ((walker ())#structure { Scope.frames = []; modules = Map.empty (module String) } ast
        : structure);
    if !changed then pass ()
  in
  pass ();
  List.dedup_and_sort !lines ~compare:Int.compare

let read content =
  let lexemes = Prose_lexemes.of_source content in
  let literals =
    List.filter lexemes ~f:(fun (l : Prose_lexemes.t) ->
        match l.kind with String_literal -> true | Comment -> false)
  in
  let claim_lines =
    List.filter_map literals ~f:(fun l ->
        let lower = String.lowercase l.text in
        if
          List.exists subject_phrases ~f:(fun pattern ->
              String.is_substring lower ~substring:pattern)
        then Some l.line
        else None)
  in
  let rec pinned = function
    | (a : Prose_lexemes.t) :: (b :: _ as rest) ->
        (String.equal a.text pin_key && String.equal b.text pin_value) || pinned rest
    | _ -> false
  in
  let markers =
    List.filter_map lexemes ~f:(fun (l : Prose_lexemes.t) ->
        match l.kind with
        | Comment when String.is_substring l.text ~substring:sentinel ->
            Some (l.line, parse_marker l.text)
        | Comment | String_literal -> None)
  in
  { claim_lines = claim_lines @ code_claim_lines content; pinned = pinned literals; markers }

(* The source's disposition, or why it has none. [None] for a source outside the subject. *)
let judge reading =
  let malformed =
    List.filter_map reading.markers ~f:(fun (line, m) ->
        match m with
        | Error why -> Some (Printf.sprintf "line %d: malformed marker: %s" line why)
        | Ok _ -> None)
  in
  let well_formed = List.filter_map reading.markers ~f:(fun (_, m) -> Result.ok m) in
  match (malformed, reading.claim_lines, reading.pinned, well_formed) with
  | _ :: _, _, _, _ -> Some (Error (String.concat ~sep:"; " malformed))
  | [], [], _, [] -> None
  | [], [], _, _ :: _ -> Some (Error "a marker, but no bitwise claim for it to dispose of")
  | [], _ :: _, true, [] -> Some (Ok Pinned)
  | [], _ :: _, true, _ :: _ -> Some (Error "both a pin and a marker: drop the marker")
  | [], _ :: _, false, [ (kind, reason) ] -> Some (Ok (Marked (kind, reason)))
  | [], _ :: _, false, _ :: _ :: _ -> Some (Error "more than one marker")
  | [], line :: _, false, [] ->
      Some
        (Error
           (Printf.sprintf
              "line %d claims bitwise equality, but the source neither pins \
               OCANNL_CC_BACKEND_FP_CONTRACT=off nor carries a marker"
              line))

let judge_text content = judge (read content)

(* The reader against synthetic sources: each direction, and the nearest text it must not take. *)
let cases () =
  let claim = "let () = p \"tiled matches serial bitwise\" true\n" in
  let pin = "let () = Unix.putenv \"OCANNL_CC_BACKEND_FP_CONTRACT\" \"off\"\n" in
  let marker kind reason = Printf.sprintf "(* %s %s -- %s *)\n" sentinel kind reason in
  let refused src = match judge_text src with Some (Error _) -> true | _ -> false in
  List.iter Fp_contract_pin_scan_cases.cases ~f:(fun (name, subject, source) ->
      p ("case: code subject -- " ^ name)
        (Test_scan.Scan_attempt.attempted ~what:"fp-contract code subjects" ~name ~default:false
           (fun () -> Bool.equal subject (refused source))));
  p "case: a pinned claim is listed as pinned"
    (match judge_text (pin ^ claim) with Some (Ok Pinned) -> true | _ -> false);
  p "case: a marked claim is listed with its kind"
    (match judge_text (marker "same-shape" "one kernel, twice" ^ claim) with
    | Some (Ok (Marked ("same-shape", "one kernel, twice"))) -> true
    | _ -> false);
  p "case: a claim with neither pin nor marker is refused" (refused claim);
  p "case: a claim in capitals is a claim too"
    (refused "let () = p \"tiled and serial agree BIT-FOR-BIT\" true\n");
  p "case: a pin to another mode is no pin"
    (refused ("let () = Unix.putenv \"OCANNL_CC_BACKEND_FP_CONTRACT\" \"fast\"\n" ^ claim));
  p "case: a marker of an unknown kind is refused"
    (refused (marker "trust-me" "it is fine" ^ claim));
  p "case: a one-word reason is refused" (refused (marker "gated" "flags" ^ claim));
  p "case: a sentinel that does not open its comment is refused"
    (refused ("(* see " ^ sentinel ^ " gated -- reads the flags *)\n" ^ claim));
  p "case: two markers are refused"
    (refused (marker "gated" "reads the flags" ^ marker "not-cc" "GPU legs only" ^ claim));
  p "case: a pin and a marker together are refused"
    (refused (pin ^ marker "gated" "reads the flags" ^ claim));
  p "case: a marker with no claim is refused" (refused (marker "gated" "reads the flags"));
  p "case: the word in a comment is no claim"
    (Option.is_none (judge_text "(* the two agree bitwise *)\nlet () = p \"tiled matches\" true\n"))

(* The sources arrive as a `@<path>` response file, as verdict_ratchet's do; see
   [Test_scan.Scan_argv]. *)
let argv = Test_scan.Scan_argv.expand Stdlib.Sys.argv

let () =
  if Array.length argv < 2 then (
    eprintf "Usage: %s <workspace_root> <source...>\n" argv.(0);
    Stdlib.exit 2);
  let base = Dune.base_dir argv.(1) in
  let arguments =
    Array.to_list (Array.subo argv ~pos:2)
    |> List.map ~f:(fun path -> (Dune.repo_relative base path, path))
  in
  let on_disk = Map.of_alist_reduce (module String) arguments ~f:(fun first _ -> first) in
  claim_slots :=
    load_claim_slots (In_channel.read_all (Map.find_exn on_disk "test/support/verdict.ml"));
  let sources =
    Sources.sources_among (List.map arguments ~f:fst)
    |> List.filter ~f:Test_scan.Ll_test_scan.test_source
  in
  let read_sources, unlexed =
    List.partition_map sources ~f:(fun source ->
        match
          Test_scan.Scan_attempt.run (fun () ->
              judge_text (In_channel.read_all (Map.find_exn on_disk source)))
        with
        | Scanned verdict -> First (source, verdict)
        | Unparsed why -> Second (source ^ ": does not parse: " ^ why)
        | Raised why -> Second (source ^ ": scanner raised: " ^ why))
  in
  let listed =
    List.filter_map read_sources ~f:(fun (source, v) -> Option.map v ~f:(fun v -> (source, v)))
  in
  let entries =
    List.filter_map listed ~f:(fun (s, v) -> Result.ok v |> Option.map ~f:(fun d -> (s, d)))
  in
  let refusals =
    List.filter_map listed ~f:(fun (s, v) ->
        match v with Error why -> Some (s ^ ": " ^ why) | Ok _ -> None)
  in
  eprintf "fp-contract pin scan: %d sources read, %d listed (not part of the golden)\n"
    (List.length read_sources) (List.length listed);
  List.iter (unlexed @ refusals) ~f:(eprintf "  refused %s\n");
  cases ();
  p_empty "every test source lexes" ~over:sources unlexed;
  p_exists "the walk read the arrayjit package's test sources too" read_sources ~f:(fun (path, _) ->
      String.is_prefix path ~prefix:"arrayjit/test/");
  p_exists "some listed source pins the mode in-process" entries ~f:(fun (_, d) ->
      match d with Pinned -> true | Marked _ -> false);
  p_exists "some listed source carries a marker" entries ~f:(fun (_, d) ->
      match d with Marked _ -> true | Pinned -> false);
  p_empty
    "every test source claiming bitwise equality pins cc_backend_fp_contract=off or carries one \
     well-formed marker, and every marker disposes of a claim"
    ~over:listed refusals;
  printf "\nKinds:\n";
  List.iter kinds ~f:(fun (kind, meaning) -> printf "%s -- %s\n" kind meaning);
  printf "\nTest sources with bitwise equality subjects:\n";
  List.iter entries ~f:(fun (source, d) ->
      match d with
      | Pinned -> printf "%s -- pinned off\n" source
      | Marked (kind, reason) -> printf "%s -- %s: %s\n" source kind reason);
  Test_scan.Refusal_control_manifest.print "fp_contract_pin_scan.ml"
