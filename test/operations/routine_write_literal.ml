(* gh-ocannl-2104: a routine write to a host-constant literal is refused at compile.

   A literal's values are declared forever equal to its registered host-init data, and a hoisted
   [Stage] packs those registered values at link time. Before the refusal, a [%cd] writer of a
   literal above [limit_constant_fill_size] (Reshape-backed, so no fetch made it fetch-defined)
   compiled, and a consumer whose read of the literal was hoisted read the old values while the
   literal's buffer held the written ones: [x *. 2] came out as [2 (i + 1)] beside a materialized [2
   (i + 101)]. A literal at most that limit long was refused already, as fetch-defined
   (gh-ocannl-1876); it now names the literal the same way at both sizes.

   The reference is the same writer-then-consumer pair over a data-backed twin, whose values vary
   with every cell and whose written values ([i + 101]) are clear of the registered ones. The
   literal's writer is refused in both compile orders (before the hoisted consumer, and from the
   consumer's context after it) and through a slice view of it, and a refusal leaves the hoisted
   consumer reading the literal's own values. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module IDX = Train.IDX
open Verdict.Claims

let stage x opt =
  let axis =
    List.hd_exn
      (List.find_exn (Ll_test.nest_paths opt.Ir.Low_level.llc) ~f:(fun p -> List.length p = 1))
  in
  Ir.Schedule.apply
    [
      Ir.Schedule.Stage
        {
          source = x.Tensor.value;
          tile_loops = [ axis ];
          shared = false;
          cooperative = None;
          hoisted = true;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        };
    ]
    opt

let refusal action =
  match action () with _ -> None | exception Utils.User_error message -> Some message

let names_literal ~node message =
  String.is_substring message ~substring:"gh-ocannl-2104"
  && String.is_substring message ~substring:node
  && String.is_substring message ~substring:"Operation.init"

let values n ~f = Array.init n ~f:(fun i -> f (Float.of_int (i + 1)))

(* The writer then the consumer over [x], run in that order; [hoisted] stages the consumer's read of
   [x] out of the routine. *)
let write_then_read ~tag ~hoisted x =
  let%cd write = x =: x + !.100.0 in
  let%op y = x *. 2 in
  let ctx, writer = Context.compile ~name:("w_" ^ tag) (Context.auto ()) write IDX.empty in
  let ctx, reader =
    Context.compile ~name:("r_" ^ tag)
      ~lowered_transform:(fun o -> [ (if hoisted then stage x o else o) ])
      ctx (Train.forward y) IDX.empty
  in
  let ctx = Context.run ctx writer in
  let ctx = Context.run ctx reader in
  Context.get_values ctx y.Tensor.value

let () =
  List.iter [ 2; 32 ] ~f:(fun n ->
      (* The data-backed twin: mutable, so the write lands and the consumer reads it. *)
      let d =
        NTDSL.init ~l:"twin" ~prec:Ir.Ops.single ~o:[ n ]
          ~f:(fun idcs -> Float.of_int (idcs.(0) + 1))
          ()
      in
      let reference = write_then_read ~tag:(Printf.sprintf "twin_%d" n) ~hoisted:false d in
      p_all2
        (Printf.sprintf "%d: the data-backed twin's consumer reads the written values" n)
        reference
        (values n ~f:(fun v -> 2. *. (v +. 100.)))
        ~f:Float.equal;
      List.iter [ false; true ] ~f:(fun hoisted ->
          let tag = Printf.sprintf "%d_%s" n (if hoisted then "hoisted" else "materialized") in
          let x = TDSL.ndarray (values n ~f:Fn.id) ~label:[ "literal" ] ~output_dims:[ n ] () in
          Train.set_materialized x.Tensor.value;
          (match write_then_read ~tag ~hoisted x with
          | ys ->
              p (tag ^ ": a writer of the literal is refused at compile") false;
              (* What the refusal prevents, when it is missing: the consumer against the twin. *)
              p_all2
                (tag ^ ": the consumer agrees with the data-backed reference")
                ys reference ~f:Float.equal
          | exception Utils.User_error message ->
              p
                (tag ^ ": a writer of the literal is refused at compile, naming it and a remedy")
                (names_literal ~node:"literal" message));
          (* The other order: the hoisted consumer first, then the writer from its context. *)
          let%op y = x *. 2 in
          let ctx, reader =
            Context.compile ~name:("r_first_" ^ tag)
              ~lowered_transform:(fun o -> [ (if hoisted then stage x o else o) ])
              (Context.auto ()) (Train.forward y) IDX.empty
          in
          let%cd write = x =: x + !.100.0 in
          p
            (tag ^ ": the writer compiled after the consumer is refused too")
            (Option.value_map ~default:false ~f:(names_literal ~node:"literal")
               (refusal (fun () -> Context.compile ~name:("w_after_" ^ tag) ctx write IDX.empty)));
          let ctx = Context.run ctx reader in
          p_all2
            (tag ^ ": the consumer reads the literal's own values")
            (Context.get_values ctx y.Tensor.value)
            (values n ~f:(fun v -> 2. *. v))
            ~f:Float.equal))

(* A slice view takes after its literal parent: writing through it is refused, naming both. *)
let () =
  List.iter [ 2; 32 ] ~f:(fun n ->
      let batch_n, bindings = IDX.get_static_symbol ~static_range:2 IDX.empty in
      let parent =
        TDSL.ndarray
          (Array.init (2 * n) ~f:(fun i -> Float.of_int (i + 1)))
          ~label:[ "lit_parent" ] ~batch_dims:[ 2 ] ~output_dims:[ n ] ()
      in
      let%op view = parent @| batch_n in
      let%cd through_view = view =: !.99.0 in
      p
        (Printf.sprintf "%d: a write through a view of a literal is refused, naming both" n)
        (Option.value_map ~default:false
           ~f:(fun m -> names_literal ~node:"view" m && names_literal ~node:"lit_parent" m)
           (refusal (fun () -> Train.to_routine (Context.auto ()) bindings through_view))))
