(* gh-ocannl-1908: every benchmark model builds under the default [op_shape_schemes=check], with the
   same shapes as under [off]. [@train] passed while cifar_resnet died under [check]
   (gh-ocannl-1820), because no test built a downsampling residual model with the schemes on.

   Each conv and gpt workload in benchmarks/workloads is built from a synthetic fixture: the spec's
   metadata as gen_fixtures.py records it, and zero-valued tensors of the layouts the model reads.
   The graph is built the way its runner builds it -- bench_conv's input batch, model, loss and
   update; [Bench_gpt_model.prepare], which bench_gpt and bench_gpt_diag share -- once per mode in a
   fresh session, and every tensor's dims are compared. Nothing trains. [Nn_blocks.resnet_block] at
   stride 2 joins them: the library's own downsampling block. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module St = Safetensors
module IDX = Train.IDX
module Row = Ocannl.Row

let workloads_dir = "../../benchmarks/workloads"

let specs =
  Stdlib.Sys.readdir workloads_dir |> Array.to_list
  |> List.filter ~f:(String.is_suffix ~suffix:".json")
  |> List.sort ~compare:String.compare
  |> List.map ~f:(fun f -> Yojson.Safe.from_file (Stdlib.Filename.concat workloads_dir f))

let field spec k =
  match spec with `Assoc kv -> List.Assoc.find kv ~equal:String.equal k | _ -> None

let int_field ?default spec k =
  match (field spec k, default) with
  | Some (`Int n), _ -> n
  | None, Some d -> d
  | _ -> failwith ("bench_model_schemes: no integer field " ^ k)

let str_field ~default spec k = match field spec k with Some (`String s) -> s | _ -> default

(* gen_fixtures.py's metadata: each spec value as a string, [stage_channels] comma-joined; the keys
   the builders derive ([mode], [use_padding], [stride1], ...) default the same way when absent. *)
let metadata spec =
  match spec with
  | `Assoc kv ->
      List.filter_map kv ~f:(fun (k, v) ->
          match v with
          | `Int n -> Some (k, Int.to_string n)
          | `Float x -> Some (k, Float.to_string x)
          | `Bool b -> Some (k, Bool.to_string b)
          | `String s -> Some (k, s)
          | `List l ->
              Some
                ( k,
                  String.concat ~sep:","
                    (List.map l ~f:(function
                      | `Int n -> Int.to_string n
                      | _ -> failwith ("bench_model_schemes: non-integer list " ^ k))) )
          | _ -> None)
  | _ -> failwith "bench_model_schemes: a workload spec is not an object"

(* The batch count only sets the batch symbol's range, so two keep the fixtures small. *)
let n_batches spec = Int.min 2 (int_field spec "n_batches")

(* The tensors each model reads, in gen_fixtures.py's layouts (build_conv, build_conv_resnet,
   build_gpt). bench_conv reads the images and labels, the conv builders only metadata;
   [Bench_gpt_model.prepare] reads the token ids and the data-backed weights. *)
let fixture_tensors spec =
  let total = int_field spec "batch_size" * n_batches spec in
  match str_field ~default:"mlp" spec "model" with
  | "conv" ->
      let img = int_field spec "image_size" in
      [
        ("x", [ total; img; img; int_field ~default:1 spec "in_channels" ]);
        ("y", [ total; int_field spec "classes" ]);
      ]
  | "gpt" ->
      let d = int_field spec "d_model" and v = int_field spec "vocab" in
      let seq = int_field spec "seq_len" and dff = int_field spec "d_ff" in
      [ ("wte", [ d; v ]); ("wpe", [ seq; d ]); ("ids", [ total; seq ]); ("tgt", [ total; seq ]) ]
      @ List.concat_map
          (List.range 0 (int_field spec "n_layer"))
          ~f:(fun i ->
            let n w = Printf.sprintf "l%d_%s" i w in
            [
              (n "ffn_w1", [ dff; d ]);
              (n "ffn_b1", [ dff ]);
              (n "ffn_w2", [ d; dff ]);
              (n "ffn_b2", [ d ]);
            ])
  | model -> failwith ("bench_model_schemes: no fixture for model " ^ model)

(* A safetensors file of F32 zeros: the 8-byte header length, the JSON header padded to an 8-byte
   boundary (so the payloads map rather than decode), the payloads back to back. *)
let write_fixture ~meta tensors =
  let path = Stdlib.Filename.temp_file "bench_model_schemes" ".safetensors" in
  let size = ref 0 in
  let entries =
    List.map tensors ~f:(fun (name, shape) ->
        let start = !size in
        size := start + (4 * List.fold shape ~init:1 ~f:( * ));
        ( name,
          `Assoc
            [
              ("dtype", `String "F32");
              ("shape", `List (List.map shape ~f:(fun d -> `Int d)));
              ("data_offsets", `List [ `Int start; `Int !size ]);
            ] ))
  in
  let meta = `Assoc (List.map meta ~f:(fun (k, v) -> (k, `String v))) in
  let header = Yojson.Safe.to_string (`Assoc (("__metadata__", meta) :: entries)) in
  let header = header ^ String.make ((8 - ((8 + String.length header) % 8)) % 8) ' ' in
  let len = Bytes.create 8 in
  Stdlib.Bytes.set_int64_le len 0 (Int64.of_int (String.length header));
  Stdio.Out_channel.with_file path ~binary:true ~f:(fun oc ->
      Stdio.Out_channel.output_bytes oc len;
      Stdio.Out_channel.output_string oc header;
      Stdio.Out_channel.output_string oc (String.make !size '\000'));
  path

(* bench_conv.ml's graph, up to the context: the batch slices, the model, the loss and the step's
   update code. *)
let build_conv st () =
  let batch_size = Bench_harness.meta_int st "batch_size" in
  let xs = TDSL.rebatch ~l:"xs" (St.to_ndarray st "x") () in
  let ys = TDSL.rebatch ~l:"ys" (St.to_ndarray st "y") () in
  let total = (Ir.Ndarray.dims (St.to_ndarray st "x")).(0) in
  let batch_n, _bindings = IDX.get_static_symbol ~static_range:(total / batch_size) IDX.empty in
  let%op batch_x = xs @| batch_n in
  let%op batch_y = ys @| batch_n in
  let { Bench_conv_model.model; mapping = _ } = Bench_conv_model.build st in
  let logits = model batch_x in
  let%op batch_loss =
    Nn_blocks.cross_entropy_loss ~spec:"...|v" ~normalize_by:!..batch_size () ~logits
      ~targets:batch_y
  in
  let learning_rate = TDSL.O.( !. ) (Float.of_string (Bench_harness.get_meta st "lr")) in
  ignore (Train.grad_update batch_loss : Ir.Assignments.comp);
  ignore (Train.sgd_update ~learning_rate batch_loss : Ir.Assignments.comp);
  (batch_loss, None)

(* [prepare] also makes a context and initializes the parameters in it: returned for release. *)
let build_gpt st () =
  let prepared = Bench_gpt_model.prepare ~materialize:false st in
  (prepared.batch_loss, Some prepared.ctx)

(* Every tensor of the graph, in a deterministic walk from the root: its label and its dims. The
   first [Shape.to_dims] closes the session's inference. *)
let graph_dims root =
  let seen = Hash_set.create (module Int) in
  let acc = ref [] in
  let rec walk (t : Tensor.t) =
    if not (Hash_set.mem seen t.value.id) then (
      Hash_set.add seen t.value.id;
      acc := (Ir.Tnode.label_name t.value, Array.to_list (Shape.to_dims t.shape)) :: !acc;
      List.iter t.children ~f:(fun c -> walk c.subtensor))
  in
  walk root;
  Array.of_list (List.rev !acc)

let entry_to_string (name, dims) =
  name ^ " " ^ String.concat ~sep:"x" (List.map dims ~f:Int.to_string)

let entry_equal (n1, d1) (n2, d2) = String.equal n1 n2 && List.equal Int.equal d1 d2

(* One build in a fresh session under [mode]: the graph's dims, or the shape error, and how many
   scheme applications were checked. A builder returns its graph's root and the context it made, if
   any, released once the dims are read. *)
let run ~mode build =
  Tensor.For_testing.set_op_shape_schemes (Some mode);
  Tensor.unsafe_reinitialize ();
  Exn.protect
    ~finally:(fun () -> Tensor.For_testing.set_op_shape_schemes None)
    ~f:(fun () ->
      let checked = ref 0 in
      let result =
        Tensor.For_testing.observe_schemes
          ~on_event:(function Tensor.Scheme.Checked _ -> Int.incr checked | _ -> ())
          (fun () ->
            match
              let root, ctx = build () in
              Exn.protect
                ~f:(fun () -> graph_dims root)
                ~finally:(fun () -> Option.iter ctx ~f:Context.release)
            with
            | dims -> Ok dims
            | exception Row.Shape_error (msg, _) -> Error msg
            | exception Tensor.Session_error (msg, _) -> Error msg)
      in
      (result, !checked))

let parity_case name build =
  Verdict.case (name ^ ": builds under check, with off's shapes") (fun () ->
      let check, n_checked = run ~mode:`Check build in
      let off, _ = run ~mode:`Off build in
      Result.iter_error check
        ~f:(Stdio.eprintf "(not part of the golden) %s under check: %s\n%!" name);
      Result.iter_error off ~f:(Stdio.eprintf "(not part of the golden) %s under off: %s\n%!" name);
      p "under check, the model builds without a shape error" (Result.is_ok check);
      p "control (off): the model builds" (Result.is_ok off);
      p "under check, scheme applications were checked" (n_checked > 0);
      match (check, off) with
      | Ok dims_check, Ok dims_off ->
          Stdio.printf "%s: %d tensors\n" name (Array.length dims_off);
          p_all2 "check and off give every tensor of the graph the same label and dims" dims_check
            dims_off ~f:(fun a b ->
              if not (entry_equal a b) then
                Stdio.eprintf "(not part of the golden) %s: check %s, off %s\n%!" name
                  (entry_to_string a) (entry_to_string b);
              entry_equal a b)
      | _ -> ())

let () =
  let built =
    List.filter_map specs ~f:(fun spec ->
        let model = str_field ~default:"mlp" spec "model" in
        if not (List.mem [ "conv"; "gpt" ] model ~equal:String.equal) then None
        else
          let name = str_field ~default:"?" spec "name" in
          let path = write_fixture ~meta:(metadata spec) (fixture_tensors spec) in
          let st = St.read path in
          Exn.protect
            ~finally:(fun () ->
              St.close st;
              Stdlib.Sys.remove path)
            ~f:(fun () ->
              parity_case name (if String.equal model "conv" then build_conv st else build_gpt st));
          Some model)
  in
  Verdict.case "the workload directory has conv and gpt workloads" (fun () ->
      p_exists "a conv workload was built" built ~f:(String.equal "conv");
      p_exists "a gpt workload was built" built ~f:(String.equal "gpt"))

(* [Nn_blocks.resnet_block] at stride 2: a strided 3x3 main path and a strided 1x1 downsample
   shortcut, with and without a channel change. *)
let () =
  List.iter [ Some 8; None ] ~f:(fun out_channels ->
      let name =
        "resnet_block stride 2"
        ^ Option.value_map out_channels ~default:"" ~f:(Printf.sprintf ", %d out channels")
      in
      parity_case name (fun () ->
          let block = Nn_blocks.resnet_block ~label:[ "rb" ] ~stride:2 ?out_channels () in
          let x =
            TDSL.range_of_shape ~label:[ "input" ] ~batch_dims:[ 2 ] ~output_dims:[ 8; 12; 3 ] ()
          in
          let%op loss = block ~train_step:None x ++ "...|... => 0" in
          (loss, None)))
